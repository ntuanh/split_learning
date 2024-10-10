import pika
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

address = "0.0.0.0"


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_part2 = ModelPart2().to(device)
optimizer2 = optim.SGD(model_part2.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

credentials = pika.PlainCredentials('guest', 'guest')
connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))


def send_gradient(gradient):
    channel = connection.channel()

    channel.queue_declare(queue='gradient_queue', durable=False)

    message = pickle.dumps(gradient.detach().cpu().numpy())

    channel.basic_publish(
        exchange='',
        routing_key='gradient_queue',
        body=message
    )
    print(" [x] Sent gradient")

    # connection.close()


def on_message_callback(ch, method, properties, body):
    print(" [x] Received intermediate output")
    received_data = pickle.loads(body)
    intermediate_output_numpy = received_data["data"]
    labels = received_data["label"].to(device)
    intermediate_output = torch.tensor(intermediate_output_numpy, requires_grad=True).to(device)

    model_part2.train()
    optimizer2.zero_grad()
    output = model_part2(intermediate_output)
    loss = criterion(output, labels)
    print(f" [x] Loss: {loss.item()}")
    intermediate_output.retain_grad()
    loss.backward()
    optimizer2.step()

    gradient = intermediate_output.grad
    send_gradient(gradient)

    ch.basic_ack(delivery_tag=method.delivery_tag)


def train_on_device_2():
    channel = connection.channel()
    channel.queue_declare(queue='intermediate_queue', durable=False)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='intermediate_queue',
                          on_message_callback=on_message_callback)

    print(' [*] Waiting for intermediate output. To exit press CTRL+C')

    channel.start_consuming()


if __name__ == "__main__":
    train_on_device_2()
