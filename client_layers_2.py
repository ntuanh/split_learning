import pika
import uuid
import pickle
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser(description="Split learning framework")
parser.add_argument('--id', type=int, required=True, help='ID of client')

args = parser.parse_args()
assert args.id is not None, "Must provide id for client."

layer_id = 2
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


class ModelPart2(nn.Module):
    def __init__(self, ResBlock=Bottleneck, layer_list=[3, 4, 6, 3], num_classes=10):
        super(ModelPart2, self).__init__()
        self.in_channels = 64

        self.layer1 = self._make_layer(ResBlock, planes=64)
        self.layer2 = identity_layers(ResBlock, layer_list[0], planes=64)
        self.layer3 = self._make_layer(ResBlock, planes=128, stride=2)
        self.layer4 = identity_layers(ResBlock, layer_list[1], planes=128)
        self.layer5 = self._make_layer(ResBlock, planes=256, stride=2)
        self.layer6 = identity_layers(ResBlock, layer_list[2], planes=256)
        self.layer7 = self._make_layer(ResBlock, planes=512, stride=2)
        self.layer8 = identity_layers(ResBlock, layer_list[3], planes=512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
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


model_part2 = ModelPart2().to(device)
optimizer2 = optim.SGD(model_part2.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


class RpcClient:
    def __init__(self):
        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(queue=self.callback_queue,
                                   on_message_callback=self.on_response,
                                   auto_ack=True)

        self.response = None
        self.corr_id = None

    def on_response(self, ch, method, props, body):
        self.response = pickle.loads(body)
        print(f"Client received: {self.response['message']}")
        action = self.response["action"]
        parameters = self.response["parameters"]

        if action == "START":
            # Read parameters and load to model
            model_part2.load_state_dict(parameters)
            # Start training
            train_on_device()
            # Stop training, then send parameters to server
            model_state_dict = model_part2.state_dict()
            data = {"action": "UPDATE", "client_id": client_id, "layer_id": layer_id, "message": "Send parameters to Server", "parameters": model_state_dict}
            self.send_to_server(data, wait=False)

    def send_to_server(self, message, wait=True):
        self.response = None
        self.corr_id = str(uuid.uuid4())

        # Send message to server
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   properties=pika.BasicProperties(
                                       reply_to=self.callback_queue,
                                       correlation_id=self.corr_id),
                                   body=pickle.dumps(message))
        if wait:
            # Wait response from server
            while self.response is None:
                self.connection.process_data_events()

    def check_process_data(self):
        self.connection.process_data_events()


client = RpcClient()
credentials = pika.PlainCredentials('dai', 'dai')
connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))


def send_gradient(gradient, trace):
    channel = connection.channel()
    to_client_id = trace[-1]
    trace.pop(-1)
    backward_queue_name = f'gradient_queue_{layer_id - 1}_{to_client_id}'
    channel.queue_declare(queue=backward_queue_name, durable=False)

    message = pickle.dumps({"data": gradient.detach().cpu().numpy(), "trace": trace})

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
        model_part2.train()
        optimizer2.zero_grad()
        # Process gradient
        method_frame, header_frame, body = channel.basic_get(queue=forward_queue_name, auto_ack=True)
        if method_frame:
            if body:
                # print("Received intermediate output")
                received_data = pickle.loads(body)
                intermediate_output_numpy = received_data["data"]
                trace = received_data["trace"]

                labels = received_data["label"].to(device)
                intermediate_output = torch.tensor(intermediate_output_numpy, requires_grad=True).to(device)

                output = model_part2(intermediate_output)
                loss = criterion(output, labels)
                print(f"Loss: {loss.item()}")
                intermediate_output.retain_grad()
                loss.backward()
                optimizer2.step()

                gradient = intermediate_output.grad
                send_gradient(gradient, trace)  # 1F1B
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
    client.send_to_server(data)
