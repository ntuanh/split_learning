import pika
import uuid
import pickle
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from src.RpcClient import RpcClient
from src.Model import FullModel

parser = argparse.ArgumentParser(description="Split learning framework")
# parser.add_argument('--id', type=int, required=True, help='ID of client')

args = parser.parse_args()

with open('../../config.yaml', 'r') as file:
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

model = FullModel()
optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

credentials = pika.PlainCredentials(username, password)
connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))


def train_on_device(trainloader):
    channel = connection.channel()
    backward_queue_name = f'gradient_queue_{layer_id}_{client_id}'
    channel.queue_declare(queue=backward_queue_name, durable=False)
    channel.basic_qos(prefetch_count=10)
    model.to(device)
    model.train()
    for (training_data, label) in tqdm(trainloader):
        training_data = training_data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(training_data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

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
