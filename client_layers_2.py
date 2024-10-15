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

layer_id = 2
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


class ModelPart2(nn.Module):
    def __init__(self, ResBlock=Bottleneck, layer_list=None, num_classes=10):
        super(ModelPart2, self).__init__()
        if layer_list is None:
            layer_list = [3, 4, 6, 3]
        self.in_channels = 64

        self.layer1 = self._make_layer(ResBlock, planes=64)
        self.layer2 = identity_layers(ResBlock, layer_list[0], planes=64)
        self.layer3 = self._make_layer(ResBlock, planes=128, stride=2)
        self.layer4 = identity_layers(ResBlock, layer_list[1], planes=128)
        self.layer5 = self._make_layer(ResBlock, planes=256, stride=2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
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


model = ModelPart2()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


credentials = pika.PlainCredentials(username, password)
connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))


def send_intermediate_output(data_id, output, labels, trace):
    channel = connection.channel()
    forward_queue_name = f'intermediate_queue_{layer_id}'
    channel.queue_declare(forward_queue_name, durable=False)
    trace.append(client_id)

    message = pickle.dumps({"data_id": data_id, "data": output.detach().cpu().numpy(), "label": labels, "trace": trace})

    channel.basic_publish(
        exchange='',
        routing_key=forward_queue_name,
        body=message
    )


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


def stop_connection():
    connection.close()


def train_on_device():
    channel = connection.channel()
    forward_queue_name = f'intermediate_queue_{layer_id - 1}'
    backward_queue_name = f'gradient_queue_{layer_id}_{client_id}'
    channel.queue_declare(queue=forward_queue_name, durable=False)
    channel.queue_declare(queue=backward_queue_name, durable=False)
    data_store = {}
    print('Waiting for intermediate output. To exit press CTRL+C')
    model.to(device)
    while True:
        # Training model
        model.train()
        optimizer.zero_grad()
        # Process gradient
        method_frame, header_frame, body = channel.basic_get(queue=backward_queue_name, auto_ack=True)
        if method_frame and body:
            received_data = pickle.loads(body)
            gradient_numpy = received_data["data"]
            gradient = torch.tensor(gradient_numpy).to(device)
            trace = received_data["trace"]
            data_id = received_data["data_id"]

            data_input = data_store.pop(data_id)
            output = model(data_input)
            data_input.retain_grad()
            output.backward(gradient=gradient, retain_graph=True)
            optimizer.step()

            gradient = data_input.grad
            send_gradient(data_id, gradient, trace)
        else:
            method_frame, header_frame, body = channel.basic_get(queue=forward_queue_name, auto_ack=True)
            if method_frame and body:
                # print("Received intermediate output")
                received_data = pickle.loads(body)
                intermediate_output_numpy = received_data["data"]
                trace = received_data["trace"]
                data_id = received_data["data_id"]
                labels = received_data["label"].to(device)

                intermediate_output = torch.tensor(intermediate_output_numpy, requires_grad=True).to(device)
                data_store[data_id] = intermediate_output

                output = model(intermediate_output)
                output = output.detach().requires_grad_(True)

                send_intermediate_output(data_id, output, labels, trace)
                # speed control
                if len(data_store) > control_count:
                    continue
        # Check training process
        if method_frame is None:
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
