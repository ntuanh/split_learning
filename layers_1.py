import pika
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import torchvision
import torchvision.transforms as transforms
import time
from tqdm import tqdm

batch_size = 256
address = "192.168.101.234"
layer_id = 1
client_id = 1


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_part1 = ModelPart1().to(device)
optimizer1 = optim.SGD(model_part1.parameters(), lr=0.01)

credentials = pika.PlainCredentials('dai', 'dai')
connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))


def send_intermediate_output(output, labels):
    channel = connection.channel()
    forward_queue_name = f'intermediate_queue_{layer_id}'
    channel.queue_declare(forward_queue_name, durable=False)

    message = pickle.dumps({"data": output.detach().cpu().numpy(), "label": labels, "trace":[client_id]})
    # TODO: change trace to append

    channel.basic_publish(
        exchange='',
        routing_key=forward_queue_name,
        body=message
    )
    # print(" [x] Sent intermediate output")

    # connection.close()


def train_on_device(trainloader):
    data_iter = iter(trainloader)
    channel = connection.channel()
    backward_queue_name = f'gradient_queue_{layer_id}_{client_id}'
    channel.queue_declare(queue=backward_queue_name, durable=False)
    count = 0
    with tqdm(total=len(trainloader), desc="Processing", unit="step") as pbar:
        while True:
            # training model
            model_part1.train()
            optimizer1.zero_grad()
            # Process gradient
            method_frame, header_frame, body = channel.basic_get(queue=backward_queue_name, auto_ack=True)
            if method_frame:
                gradient = None
                if body:
                    received_data = pickle.loads(body)
                    gradient_numpy = received_data["data"]
                    gradient = torch.tensor(gradient_numpy).to(device)
                    # print(" [x] Received gradient")
                if gradient is not None:
                    # Process backward message
                    try:
                        intermediate_output.backward(gradient)
                        optimizer1.step()
                        # print(" [x] Updated Model Part 1")
                    except:
                        # print("Something else went wrong")
                        pass
            else:
                # tqdm bar
                pbar.update(1)
                # Process forward message
                data, labels = next(data_iter)
                intermediate_output = model_part1(data.to(device))
                intermediate_output = intermediate_output.detach().requires_grad_(True)
                # Send to next layers
                send_intermediate_output(intermediate_output, labels)
                # TODO: speed control
                time.sleep(0.2)


if __name__ == "__main__":
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

    for epoch in range(5):
        print(f"Epoch {epoch + 1}")
        train_on_device(train_loader)
