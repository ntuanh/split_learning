import pika
import pickle
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from src.RpcClient import RpcClient

parser = argparse.ArgumentParser(description="Split learning framework")
parser.add_argument('--id', type=int, required=True, help='ID of client')

args = parser.parse_args()
assert args.id is not None, "Must provide id for client."

layer_id = 3
client_id = args.id
address = "192.168.101.234"
username = "dai"
password = "dai"

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


class ModelPart3(nn.Module):
    def __init__(self, ResBlock=Bottleneck, layer_list=None, num_classes=10):
        super(ModelPart3, self).__init__()
        if layer_list is None:
            layer_list = [3, 4, 6, 3]
        self.in_channels = 1024

        self.layer6 = identity_layers(ResBlock, layer_list[2], planes=256)
        self.layer7 = self._make_layer(ResBlock, planes=512, stride=2)
        self.layer8 = identity_layers(ResBlock, layer_list[3], planes=512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, ResBlock, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        return nn.Sequential(*layers)


model = ModelPart3().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


credentials = pika.PlainCredentials(username, password)
connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))


def send_gradient(data_id, gradient, trace):
    channel = connection.channel()
    to_client_id = trace[-1]
    trace.pop(-1)
    backward_queue_name = f'gradient_queue_{layer_id - 1}_{to_client_id}'
    channel.queue_declare(queue=backward_queue_name, durable=False)

    message = pickle.dumps({"data_id": data_id, "data": gradient.detach().cpu().numpy(), "trace": trace})

    channel.basic_publish(
        exchange='',
        routing_key=backward_queue_name,
        body=message
    )
    # print("Sent gradient")


def stop_connection():
    connection.close()


def train_on_device():
    channel = connection.channel()
    forward_queue_name = f'intermediate_queue_{layer_id - 1}'
    channel.queue_declare(queue=forward_queue_name, durable=False)
    print('Waiting for intermediate output. To exit press CTRL+C')

    while True:
        # Training model
        model.train()
        optimizer.zero_grad()
        # Process gradient
        method_frame, header_frame, body = channel.basic_get(queue=forward_queue_name, auto_ack=True)
        if method_frame and body:
            # print("Received intermediate output")
            received_data = pickle.loads(body)
            intermediate_output_numpy = received_data["data"]
            trace = received_data["trace"]
            data_id = received_data["data_id"]

            labels = received_data["label"].to(device)
            intermediate_output = torch.tensor(intermediate_output_numpy, requires_grad=True).to(device)

            output = model(intermediate_output)
            loss = criterion(output, labels)
            print(f"Loss: {loss.item()}")
            intermediate_output.retain_grad()
            loss.backward()
            optimizer.step()

            gradient = intermediate_output.grad
            send_gradient(data_id, gradient, trace)  # 1F1B
        # Check training process
        else:
            broadcast_queue_name = 'broadcast_queue'
            method_frame, header_frame, body = channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
            if body:
                received_data = pickle.loads(body)
                print(f"Received message from server {received_data}")
                break


if __name__ == "__main__":
    print("Client sending registration message to server...")
    data = {"action": "REGISTER", "client_id": client_id, "layer_id": layer_id, "message": "Hello from Client!"}
    client = RpcClient(client_id, layer_id, model, address, username, password, train_on_device)
    client.send_to_server(data)
