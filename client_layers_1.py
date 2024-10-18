import pika
import uuid
import pickle
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from src.RpcClient import RpcClient
from src.Model import ModelPart1

parser = argparse.ArgumentParser(description="Split learning framework")
# parser.add_argument('--id', type=int, required=True, help='ID of client')

args = parser.parse_args()

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

layer_id = 1
client_id = uuid.uuid4()
address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]
batch_size = config["learning"]["batch-size"]
lr = config["learning"]["learning-rate"]
control_count = config["learning"]["control-count"]

device = None

if torch.cuda.is_available():
    device = "cuda"
    print(f"Using device: {torch.cuda.get_device_name(device)}")
else:
    device = "cpu"
    print(f"Using device: CPU")

model = ModelPart1()
optimizer = optim.SGD(model.parameters(), lr=lr)

credentials = pika.PlainCredentials(username, password)
connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))


def send_intermediate_output(data_id, output, labels):
    channel = connection.channel()
    forward_queue_name = f'intermediate_queue_{layer_id}'
    channel.queue_declare(forward_queue_name, durable=False)

    message = pickle.dumps(
        {"data_id": data_id, "data": output.detach().cpu().numpy(), "label": labels, "trace": [client_id]})

    channel.basic_publish(
        exchange='',
        routing_key=forward_queue_name,
        body=message
    )


def train_on_device(trainloader):
    data_iter = iter(trainloader)
    channel = connection.channel()
    backward_queue_name = f'gradient_queue_{layer_id}_{client_id}'
    channel.queue_declare(queue=backward_queue_name, durable=False)
    num_forward = 0
    num_backward = 0
    end_data = False
    data_store = {}
    model.to(device)
    with tqdm(total=len(trainloader), desc="Processing", unit="step") as pbar:
        while True:
            # Training model
            model.train()
            optimizer.zero_grad()
            # Process gradient
            method_frame, header_frame, body = channel.basic_get(queue=backward_queue_name, auto_ack=True)
            if method_frame and body:
                num_backward += 1
                received_data = pickle.loads(body)
                gradient_numpy = received_data["data"]
                gradient = torch.tensor(gradient_numpy).to(device)
                data_id = received_data["data_id"]

                data_input = data_store.pop(data_id)
                output = model(data_input)
                output.backward(gradient=gradient, retain_graph=True)
                optimizer.step()
            else:
                # speed control
                if len(data_store) > control_count:
                    continue
                # Process forward message
                try:
                    training_data, labels = next(data_iter)
                    training_data = training_data.to(device)
                    data_id = uuid.uuid4()
                    data_store[data_id] = training_data
                    intermediate_output = model(training_data)
                    intermediate_output = intermediate_output.detach().requires_grad_(True)

                    # Send to next layers
                    num_forward += 1
                    # tqdm bar
                    pbar.update(1)

                    send_intermediate_output(data_id, intermediate_output, labels)

                except StopIteration:
                    end_data = True
            if end_data and (num_forward == num_backward):
                # Finish epoch training, send notify to server
                print("Finish training!")
                training_data = {"action": "NOTIFY", "client_id": client_id, "layer_id": layer_id,
                                 "message": "Finish training!"}
                client.send_to_server(training_data)

                while True:  # Wait for broadcast
                    broadcast_queue_name = 'broadcast_queue'
                    method_frame, header_frame, body = channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
                    if body:
                        received_data = pickle.loads(body)
                        print(f"Received message from server {received_data}")
                        break
                break


if __name__ == "__main__":
    print("Client sending registration message to server...")
    # Read and load dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False)

    data = {"action": "REGISTER", "client_id": client_id, "layer_id": layer_id, "message": "Hello from Client!"}
    client = RpcClient(client_id, layer_id, model, address, username, password, train_on_device, train_loader)
    client.send_to_server(data)
    client.wait_response()
