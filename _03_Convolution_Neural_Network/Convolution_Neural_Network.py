# 在该文件NeuralNetwork类中定义你的模型 
# 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型

import os

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")
os.system("sudo pip3 install tqdm")
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import collections
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
###声明使用gpu进行训练
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("使用GPU进行训练")
else:
    device = torch.device("cpu")
    print("使用cpu进行训练")
###定义好数据集的类型
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

])
####定义好CIFAR-10的标签
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
###定义好vgg的配置,字典类型
cfg ={
    'VGG16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

###编写网络
class NeuralNetwork(nn.Module):
    def __init__(self,vgg_name):
        super(NeuralNetwork,self).__init__()##初始化父类
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512,10)###全连接层
    def forward(self,x):
        x= self.features(x)
        x = x.view(x.size(0),-1)
        x=self.classifier(x)
        return x
    def _make_layers(self,cfg):
        layers = []
        in_channels =3
        for x in cfg:
            if x =='M':
                layers +=[nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                layers +=[nn.Conv2d(in_channels,x,kernel_size=3,padding=1),
                          nn.BatchNorm2d(x),
                          nn.ReLU(inplace=True)]
                in_channels =x
        layers += [nn.AvgPool2d(kernel_size=1,stride=1)]
        return nn.Sequential(*layers)

def train():
    LR = 0.001
    EPOCHS = 40
    BATCHSIZE = 100
    trainset , testset , trainloader,testloader=read_data()




    net4 = NeuralNetwork('VGG16').to(device)
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
    torch.save(net4, '../../pth/model.pth')

def read_data():
    # 这里可自行修改数据预处理，batch大小也可自行调整
    # 保持本地训练的数据读取和这里一致
    dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=False, transform=transform)
    dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False, transform=transform)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=256, shuffle=False)
    return dataset_train, dataset_val, data_loader_train, data_loader_val

def main():

    model = NeuralNetwork('VGG16') # 若有参数则传入参数
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    # model.load_state_dict(torch.load(parent_dir + '/pth/model.pth'))
    state_dict = torch.load("../../pth/model.pth")  # 加载模型权重文件
    model.load_state_dict(state_dict)  # 使用加载的权重更新模型参数
    return model

    