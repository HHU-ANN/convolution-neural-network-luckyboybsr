import collections
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from multiprocessing import freeze_support
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # 可以根据需要更改设备号，比如cuda:1, cuda:2等
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10('./data', train=False, download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True, num_workers=2)

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

LR = 0.001
EPOCHS = 40
BATCHSIZE = 100

if __name__ == '__main__':
    freeze_support()

    net4 = VGG('VGG16').to(device)
    mlps = [net4]

    optimizer = torch.optim.Adam([{"params": mlp.parameters()} for mlp in mlps], lr=LR)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        # 训练阶段
        with tqdm(total=len(trainloader), desc='Epoch {}/{}'.format(epoch + 1, EPOCHS)) as t:
            for img, label in trainloader:
                img, label = img.to(device), label.to(device)
                optimizer.zero_grad()
                for mlp in mlps:
                    mlp.train()
                    out = mlp(img)
                    loss = loss_function(out, label)
                    loss.backward()
                optimizer.step()
                t.update(1)
            t.close()

        pre = []
        vote_correct = 0
        mlps_correct = [0 for _ in range(len(mlps))]

        # 测试阶段
        with torch.no_grad():
            with tqdm(total=len(testloader), desc='Testing') as t:
                for img, label in testloader:
                    img, label = img.to(device), label.to(device)
                    for i, mlp in enumerate(mlps):
                        mlp.eval()
                        out = mlp(img)
                        _, prediction = torch.max(out, 1)
                        pre_num = prediction.cpu().numpy()
                        pre_num = tuple(pre_num)  # 将预测结果转换为元组形式
                        pre.append(pre_num)

                        correct = (prediction == label).sum().item()
                        mlps_correct[i] += correct

                    arr = np.array(pre)
                    pre.clear()  # 清空 pre 列表
                    num_models = len(mlps)
                    arr = np.reshape(arr, (num_models, -1))
                    result = [collections.Counter(arr[:, i]).most_common(1)[0][0] for i in range(arr.shape[1])]

                    vote_correct += np.sum(result == label.cpu().numpy())
                    t.update(1)
                t.close()

        for idx, correct in enumerate(mlps_correct):
            print("Epoch {}: VGG的正确率为: {:.4f}%".format(epoch + 1, correct * 100 / len(testloader.dataset)))
    # 保存完整模型
    torch.save(net4, 'model.pth')