import pika
import uuid
import pickle
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from src.RpcClient import RpcClient

parser = argparse.ArgumentParser(description="Split learning framework")
parser.add_argument('--id', type=int, required=True, help='ID of client')

args = parser.parse_args()
assert args.id is not None, "Must provide id for client."

batch_size = 256
layer_id = 1
client_id = args.id
address = "192.168.101.234"
username = "dai"
password = "dai"
control_count = 3

device = None

if torch.cuda.is_available():
    device = "cuda"
    print(f"Using device: {torch.cuda.get_device_name(device)}")
else:
    device = "cpu"
    print(f"Using device: CPU")


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


def identity_layers(ResBlock, blocks, planes):
    layers = []

    for i in range(blocks - 1):
        layers.append(ResBlock(planes * ResBlock.expansion, planes))

    return nn.Sequential(*layers)


class ModelPart1(nn.Module):
    def __init__(self, num_channels=3):
        super(ModelPart1, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


model = ModelPart1()
optimizer = optim.SGD(model.parameters(), lr=0.01)


credentials = pika.PlainCredentials(username, password)
connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))


def send_intermediate_output(data_id, output, labels):
    channel = connection.channel()
    forward_queue_name = f'intermediate_queue_{layer_id}'
    channel.queue_declare(forward_queue_name, durable=False)

    message = pickle.dumps({"data_id": data_id, "data": output.detach().cpu().numpy(), "label": labels, "trace": [client_id]})

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
                training_data = {"action": "NOTIFY", "client_id": client_id, "layer_id": layer_id, "message": "Finish training!"}
                client.send_to_server(training_data, wait=False)
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
