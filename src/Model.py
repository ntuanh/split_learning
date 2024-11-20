import torch
import torch.nn as nn
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

if torch.cuda.is_available():
    device = "cuda"
    print(f"Using device: {torch.cuda.get_device_name(device)}")
else:
    device = "cpu"
    print(f"Using device: CPU")

n_epochs = 5
batch_size_train = 128
batch_size_test = 100
learning_rate = 0.01
momentum = 0.5
log_interval = 10


class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.BatchNorm2d(64)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer5 = nn.BatchNorm2d(64)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer8 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.layer9 = nn.BatchNorm2d(128)
        self.layer10 = nn.ReLU()
        self.layer11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.layer12 = nn.BatchNorm2d(128)
        self.layer13 = nn.ReLU()
        self.layer14 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer15 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.layer16 = nn.BatchNorm2d(256)
        self.layer17 = nn.ReLU()
        self.layer18 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer19 = nn.BatchNorm2d(256)
        self.layer20 = nn.ReLU()
        self.layer21 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer22 = nn.BatchNorm2d(256)
        self.layer23 = nn.ReLU()
        self.layer24 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer25 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.layer26 = nn.BatchNorm2d(512)
        self.layer27 = nn.ReLU()
        self.layer28 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer29 = nn.BatchNorm2d(512)
        self.layer30 = nn.ReLU()
        self.layer31 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer32 = nn.BatchNorm2d(512)
        self.layer33 = nn.ReLU()
        self.layer34 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer35 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer36 = nn.BatchNorm2d(512)
        self.layer37 = nn.ReLU()
        self.layer38 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer39 = nn.BatchNorm2d(512)
        self.layer40 = nn.ReLU()
        self.layer41 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer42 = nn.BatchNorm2d(512)
        self.layer43 = nn.ReLU()
        self.layer44 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer45 = nn.Flatten(1, -1)
        self.layer46 = nn.Dropout(0.5)
        self.layer47 = nn.Linear(1 * 1 * 512, 4096)
        self.layer48 = nn.ReLU()
        self.layer49 = nn.Dropout(0.5)
        self.layer50 = nn.Linear(4096, 4096)
        self.layer51 = nn.ReLU()
        self.layer52 = nn.Linear(4096, 10)

    def forward(self, input0):
        out1 = self.layer1(input0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        out12 = self.layer12(out11)
        out13 = self.layer13(out12)
        out14 = self.layer14(out13)
        out15 = self.layer15(out14)
        out16 = self.layer16(out15)
        out17 = self.layer17(out16)
        out18 = self.layer18(out17)
        out19 = self.layer19(out18)
        out20 = self.layer20(out19)
        out21 = self.layer21(out20)
        out22 = self.layer22(out21)
        out23 = self.layer23(out22)
        out24 = self.layer24(out23)
        out25 = self.layer25(out24)
        out26 = self.layer26(out25)
        out27 = self.layer27(out26)
        out28 = self.layer28(out27)
        out29 = self.layer29(out28)
        out30 = self.layer30(out29)
        out31 = self.layer31(out30)
        out32 = self.layer32(out31)
        out33 = self.layer33(out32)
        out34 = self.layer34(out33)
        out35 = self.layer35(out34)
        out36 = self.layer36(out35)
        out37 = self.layer37(out36)
        out38 = self.layer38(out37)
        out39 = self.layer39(out38)
        out40 = self.layer40(out39)
        out41 = self.layer41(out40)
        out42 = self.layer42(out41)
        out43 = self.layer43(out42)
        out44 = self.layer44(out43)
        out45 = self.layer45(out44)
        out46 = self.layer46(out45)
        out47 = self.layer47(out46)
        out48 = self.layer48(out47)
        out49 = self.layer49(out48)
        out50 = self.layer50(out49)
        out51 = self.layer51(out50)
        out52 = self.layer52(out51)
        return out52


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


class FullModel(nn.Module):
    def __init__(self, model, cut_layers):
        super(FullModel, self).__init__()
        self.full_model = []
        for i in range(len(cut_layers) + 1):
            if i == 0:
                self.full_model.append(nn.Sequential(*nn.ModuleList(model.children())[:cut_layers[0]]))
            elif i == len(cut_layers):
                self.full_model.append(nn.Sequential(*nn.ModuleList(model.children())[cut_layers[i - 1]:]))
            else:
                self.full_model.append(
                    nn.Sequential(*nn.ModuleList(model.children())[cut_layers[i - 1]:cut_layers[i]]))

    def forward(self, x):
        for sub_model in self.full_model:
            x = sub_model(x)
        return x


def test(model_name, cut_layers, logger):
    klass = globals().get(model_name)
    if klass is None:
        raise ValueError(f"Class '{model_name}' does not exist.")
    model = klass()
    models = FullModel(model, cut_layers)

    for i, sub_model in enumerate(models.full_model):
        part_i_state_dict = torch.load(f'{model_name}_{i + 1}.pth', weights_only=False)
        sub_model.load_state_dict(part_i_state_dict)
    # evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in tqdm(test_loader):
        output = models(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100.0 * correct / len(test_loader.dataset)))
    logger.log_info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100.0 * correct / len(test_loader.dataset)))
