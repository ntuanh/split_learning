import time
import pika
import uuid
import pickle
import argparse
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import src.Log
from src.RpcClient import RpcClient
from Model import ModelPart1

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
validation = config["learning"]["validation"]

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
channel = connection.channel()


def send_intermediate_output(data_id, output, labels, test=False):
    forward_queue_name = f'intermediate_queue_{layer_id}'
    channel.queue_declare(forward_queue_name, durable=False)

    message = pickle.dumps(
        {"data_id": data_id, "data": output.detach().cpu().numpy(), "label": labels, "trace": [client_id], "test": test}
    )

    channel.basic_publish(
        exchange='',
        routing_key=forward_queue_name,
        body=message
    )


def train_on_device(trainloader, testloader=None):
    data_iter = iter(trainloader)
    backward_queue_name = f'gradient_queue_{layer_id}_{client_id}'
    channel.queue_declare(queue=backward_queue_name, durable=False)
    channel.basic_qos(prefetch_count=10)
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
                break

    # validation
    num_forward = 0
    num_backward = 0
    all_labels = np.array([])
    all_vals = np.array([])
    if testloader:
        for (testing_data, labels) in testloader:
            testing_data = testing_data.to(device)
            data_id = uuid.uuid4()
            intermediate_output = model(testing_data)

            # Send to next layers
            num_forward += 1

            send_intermediate_output(data_id, intermediate_output, labels, test=True)

        while True:
            method_frame, header_frame, body = channel.basic_get(queue=backward_queue_name, auto_ack=True)
            if method_frame and body:
                num_backward += 1
                received_data = pickle.loads(body)
                test_data = received_data["data"]

                all_labels = np.append(all_labels, test_data[0])
                all_vals = np.append(all_vals, test_data[1])

            if num_forward == num_backward:
                break

        notify_data = {"action": "NOTIFY", "client_id": client_id, "layer_id": layer_id,
                       "message": "Finish training!", "validate": [all_labels, all_vals]}
    else:
        notify_data = {"action": "NOTIFY", "client_id": client_id, "layer_id": layer_id,
                       "message": "Finish training!", "validate": None}

    # Finish epoch training, send notify to server
    src.Log.print_with_color("[>>>] Finish training!", "red")
    client.send_to_server(notify_data)

    broadcast_queue_name = f'reply_{client_id}'
    while True:  # Wait for broadcast
        method_frame, header_frame, body = channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
        if body:
            received_data = pickle.loads(body)
            src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
            if received_data["action"] == "PAUSE":
                break
        time.sleep(0.5)


if __name__ == "__main__":
    src.Log.print_with_color("[>>>] Client sending registration message to server...", "red")
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

    if validation:
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False)
    else:
        test_loader = None

    data = {"action": "REGISTER", "client_id": client_id, "layer_id": layer_id, "message": "Hello from Client!"}
    client = RpcClient(client_id, layer_id, model, address, username, password, train_on_device, train_loader,
                       test_loader)
    client.send_to_server(data)
    client.wait_response()
