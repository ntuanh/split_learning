import torch
import torch.nn as nn
import time

import torchvision
import torchvision.transforms as transforms

import src.Model


model = src.Model.VGG16()
batch_size = 100

full_model = []
for sub_model in nn.Sequential(*nn.ModuleList(model.children())):
    full_model.append(sub_model)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2)

data_size = []
forward_time = []

train_data = None

for (data, target) in test_loader:
    train_data = data
    break

data = train_data

# Forward
for sub_model in full_model:
    start = time.time_ns()
    time.time()
    sub_model.train()
    data = sub_model(data)
    data_size.append(data.nelement() * data.element_size())
    end = time.time_ns()
    forward_time.append(end-start)

print(f"List of forward tranining time = {forward_time} nano second")
print(f"List of data size = {data_size} bytes")
