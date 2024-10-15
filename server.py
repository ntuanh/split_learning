import os
import pika
import pickle
import argparse

import torch

import requests
from requests.auth import HTTPBasicAuth
from src.test import test

parser = argparse.ArgumentParser(description="Split learning framework with controller.")

parser.add_argument('--topo', type=int, nargs='+', required=True, help='List of client topo, example: --topo 2 3')

args = parser.parse_args()

total_clients = args.topo
filename = "resnet_model"
address = "192.168.101.234"
username = "dai"
password = "dai"


class Server:
    def __init__(self):
        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
        self.channel = self.connection.channel()

        self.channel.queue_declare(queue='rpc_queue')
        self.channel.queue_declare('broadcast_queue', durable=False)

        self.total_clients = total_clients
        self.current_clients = [0 for _ in range(len(total_clients))]
        self.first_layer_clients = 0
        self.responses = {}  # Save response

        self.all_model_parameters = [[] for _ in range(len(total_clients))]

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)
        print(f"Server is waiting for {self.total_clients} clients.")

    def on_request(self, ch, method, props, body):
        message = pickle.loads(body)
        routing_key = props.reply_to
        action = message["action"]
        client_id = message["client_id"]
        layer_id = message["layer_id"]

        if action == "REGISTER":
            print(f"Received message from client: {message}")
            # Save messages from clients
            self.responses[routing_key] = message
            self.current_clients[layer_id - 1] += 1

            # If consumed all clients
            if self.current_clients == self.total_clients:
                print("All clients are connected. Sending notifications.")
                self.notify_clients(ch)
                self.current_clients = [0 for _ in range(len(total_clients))]
        elif action == "NOTIFY":
            print(f"Received message from client: {message}")
            if layer_id == 1:
                self.first_layer_clients += 1

            if self.first_layer_clients == self.total_clients[0]:
                print("Received finish training notification")
                self.stop_training_round(ch)
                for _ in range(sum(self.total_clients[1:])):
                    self.send_to_broadcast()
        elif action == "UPDATE":
            # Save client's model parameters
            model_state_dict = message["parameters"]
            self.current_clients[layer_id - 1] += 1
            self.all_model_parameters[layer_id - 1].append(model_state_dict)

            # If consumed all client's parameters
            if self.current_clients == self.total_clients:
                print("Collected all parameters.")
                self.avg_all_parameters()
                self.current_clients = [0 for _ in range(len(total_clients))]
                self.all_model_parameters = [[] for _ in range(len(total_clients))]
                # Test
                test()
                # TODO: Start a new training round

        # Ack the message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def notify_clients(self, channel):
        # Send message to clients when consumed all clients
        for routing_key in self.responses:
            layer = self.responses[routing_key]["layer_id"]
            # Read parameters file
            filepath = f'{filename}_{layer}.pth'
            state_dict = None
            if os.path.exists(filepath):
                state_dict = torch.load(filepath, weights_only=False)
                print("Model loaded successfully.")
            else:
                print(f"File {filepath} does not exist.")

            response = {"action": "START",
                        "message": "Server accept the connection!",
                        "parameters": state_dict}

            channel.basic_publish(exchange='',
                                  routing_key=routing_key,
                                  properties=pika.BasicProperties(),
                                  body=pickle.dumps(response))
            print(f"Sent notification to client {routing_key}")

    def stop_training_round(self, channel):
        # Send message to clients when consumed all clients
        for routing_key in self.responses:
            response = {"action": "STOP",
                        "message": "Stop training and please send your parameters",
                        "parameters": None}

            channel.basic_publish(exchange='',
                                  routing_key=routing_key,
                                  properties=pika.BasicProperties(),
                                  body=pickle.dumps(response))
            print(f"Send stop training request to clients {routing_key}")

    def start(self):
        self.channel.start_consuming()

    def send_to_broadcast(self):
        broadcast_channel = self.connection.channel()
        broadcast_queue_name = 'broadcast_queue'
        broadcast_channel.queue_declare(broadcast_queue_name, durable=False)

        message = pickle.dumps({"action": "STOP",
                                "message": "Stop training and please send your parameters",
                                "parameters": None})
        broadcast_channel.basic_publish(
            exchange='',
            routing_key=broadcast_queue_name,
            body=message
        )

    def avg_all_parameters(self):
        # Average all client parameters
        for layer, state_dicts in enumerate(self.all_model_parameters):
            avg_state_dict = {}
            num_models = len(state_dicts)

            for key in state_dicts[0].keys():
                if state_dicts[0][key].dtype == torch.long:
                    avg_state_dict[key] = state_dicts[0][key].float()
                else:
                    avg_state_dict[key] = state_dicts[0][key].clone()

                for i in range(1, num_models):
                    if state_dicts[i][key].dtype == torch.long:
                        avg_state_dict[key] += state_dicts[i][key].float()
                    else:
                        avg_state_dict[key] += state_dicts[i][key]

                avg_state_dict[key] /= num_models

                if state_dicts[0][key].dtype == torch.long:
                    avg_state_dict[key] = avg_state_dict[key].long()
            # Save to files
            torch.save(avg_state_dict, f'{filename}_{layer + 1}.pth')


def delete_old_queues():
    url = f'http://{address}:15672/api/queues'
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        queues = response.json()

        credentials = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
        http_channel = connection.channel()

        for queue in queues:
            queue_name = queue['name']
            if queue_name.startswith("amq.gen-"):
                try:
                    http_channel.queue_delete(queue=queue_name)
                    print(f"Queue '{queue_name}' deleted.")
                except Exception as e:
                    print(f"Failed to delete queue '{queue_name}': {e}")
            else:
                try:
                    http_channel.queue_purge(queue=queue_name)
                    print(f"Queue '{queue_name}' deleted.")
                except Exception as e:
                    print(f"Failed to purge queue '{queue_name}': {e}")

        connection.close()
        return True
    else:
        print(f"Failed to fetch queues from RabbitMQ Management API. Status code: {response.status_code}")
        return False

# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.batch_norm1 = nn.BatchNorm2d(out_channels)
#
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
#         self.batch_norm2 = nn.BatchNorm2d(out_channels)
#
#         self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
#         self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)
#
#         self.i_downsample = i_downsample
#         self.stride = stride
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         identity = x.clone()
#         x = self.relu(self.batch_norm1(self.conv1(x)))
#
#         x = self.relu(self.batch_norm2(self.conv2(x)))
#
#         x = self.conv3(x)
#         x = self.batch_norm3(x)
#
#         # downsample if needed
#         if self.i_downsample is not None:
#             identity = self.i_downsample(identity)
#         # add identity
#         x += identity
#         x = self.relu(x)
#
#         return x

#
# def identity_layers(ResBlock, blocks, planes):
#     layers = []
#
#     for i in range(blocks - 1):
#         layers.append(ResBlock(planes * ResBlock.expansion, planes))
#
#     return nn.Sequential(*layers)
#
#
# class ModelPart1(nn.Module):
#     def __init__(self, num_channels=3):
#         super(ModelPart1, self).__init__()
#         self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.batch_norm1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU()
#         self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.batch_norm1(x)
#         x = self.relu(x)
#         x = self.max_pool(x)
#         return x
#
# class ModelPart2(nn.Module):
#     def __init__(self, ResBlock=Bottleneck, layer_list=None, num_classes=10):
#         super(ModelPart2, self).__init__()
#         if layer_list is None:
#             layer_list = [3, 4, 6, 3]
#         self.in_channels = 64
#
#         self.layer1 = self._make_layer(ResBlock, planes=64)
#         self.layer2 = identity_layers(ResBlock, layer_list[0], planes=64)
#         self.layer3 = self._make_layer(ResBlock, planes=128, stride=2)
#         self.layer4 = identity_layers(ResBlock, layer_list[1], planes=128)
#         self.layer5 = self._make_layer(ResBlock, planes=256, stride=2)
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         return x
#
#     def _make_layer(self, ResBlock, planes, stride=1):
#         ii_downsample = None
#         layers = []
#
#         if stride != 1 or self.in_channels != planes * ResBlock.expansion:
#             ii_downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(planes * ResBlock.expansion)
#             )
#
#         layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
#         self.in_channels = planes * ResBlock.expansion
#
#         return nn.Sequential(*layers)
#
# class ModelPart3(nn.Module):
#     def __init__(self, ResBlock=Bottleneck, layer_list=None, num_classes=10):
#         super(ModelPart3, self).__init__()
#         if layer_list is None:
#             layer_list = [3, 4, 6, 3]
#         self.in_channels = 1024
#
#         self.layer6 = identity_layers(ResBlock, layer_list[2], planes=256)
#         self.layer7 = self._make_layer(ResBlock, planes=512, stride=2)
#         self.layer8 = identity_layers(ResBlock, layer_list[3], planes=512)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)
#
#     def forward(self, x):
#         x = self.layer6(x)
#         x = self.layer7(x)
#         x = self.layer8(x)
#         x = self.avgpool(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc(x)
#         return x
#
#     def _make_layer(self, ResBlock, planes, stride=1):
#         ii_downsample = None
#         layers = []
#
#         if stride != 1 or self.in_channels != planes * ResBlock.expansion:
#             ii_downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(planes * ResBlock.expansion)
#             )
#
#         layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
#         self.in_channels = planes * ResBlock.expansion
#
#         return nn.Sequential(*layers)
#
# class FullModel(nn.Module):
#     def __init__(self):
#         super(FullModel, self).__init__()
#         self.part1 = ModelPart1()
#         self.part2 = ModelPart2()
#         self.part3 = ModelPart3()
#
#     def forward(self, x):
#         x = self.part1(x)
#         x = self.part2(x)
#         x = self.part3(x)
#         return x
#
# def test():
#     part1_state_dict = torch.load('resnet_model_1.pth')
#     part2_state_dict = torch.load('resnet_model_2.pth')
#     part3_state_dict = torch.load('resnet_model_3.pth')
#
#     full_model = FullModel()
#
#     full_model.part1.load_state_dict(part1_state_dict)
#     full_model.part2.load_state_dict(part2_state_dict)
#     full_model.part3.load_state_dict(part3_state_dict)
#
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#
#     trainset = torchvision.datasets.CIFAR10(
#         root='./data', train=True, download=True, transform=transform_train)
#     train_loader = torch.utils.data.DataLoader(
#         trainset, batch_size=128, shuffle=True, num_workers=2)
#
#     testset = torchvision.datasets.CIFAR10(
#         root='./data', train=False, download=True, transform=transform_test)
#     test_loader = torch.utils.data.DataLoader(
#         testset, batch_size=100, shuffle=False, num_workers=2)
#
#     full_model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:
#         output = full_model(data)
#         test_loss += F.nll_loss(output, target, reduction='sum').item()
#         pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
#
#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100.0 * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    server = Server()
    delete_old_queues()
    server.start()
    print("Ok, ready!")
