import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def loss_func(pred, label):
    # pred,label: [batch_size, 24]
    # pred = pred.reshape(-1,24)
    return ((pred-label)**2).sum() / len(label)

def data_iter(batch_size, features, labels):
    features = torch.tensor(features,dtype=torch.float32)
    labels = torch.tensor(labels,dtype=torch.float32)
    num_examples = len(features)
    veh = features[:,:24].reshape(-1,4,2,3)
    inf = features[:,24:48].reshape(-1,4,2,3)
    new_features = torch.cat([veh,inf],2)
    # new_labels = labels.reshape(-1,4,2,3)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield new_features[batch_indices].permute(0,3,1,2), labels[batch_indices]

def train(net, train_iter, loss, updater): #@save
    # 将模型设置为训练模式
    net.eval()
    loss_sum = 0
    n = 0
    # 训练损失总和、训练准确度总和、样本数
    for X, y in train_iter:
    # 计算梯度并更新参数
        X, y = X.cuda(), y.cuda()
        y_hat = net(X)
        l = loss(y_hat, y)
    # 使⽤PyTorch内置的优化器和损失函数
        updater.zero_grad()
        # l.backward()
        # updater.step()
        loss_sum += l.item()
        n +=1
    print(loss_sum/n)
    # 返回训练损失和训练精度
#     return metric[0] / metric[2], metric[1] / metric[2]

if __name__ == '__main__':
  
    net = nn.Sequential(
        nn.Conv2d(3, 24, kernel_size=3,padding=1),
        nn.BatchNorm2d(24),
        nn.ReLU(),
        nn.Conv2d(24, 48, kernel_size=3,padding=1),
        nn.BatchNorm2d(48),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(48,48),
        nn.Linear(48,24),
    )
    
    df = pd.read_csv('./late-fusion-data.csv')
    features = df.values[:,:-26]
    labels = df.values[:,-24:]
    
    PATH = './linear.pth'
    net.cuda()
    net.load_state_dict(torch.load(PATH))
    batch_size, lr, num_epochs = 128, 3e-6, 10
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_iter = data_iter(batch_size, features, labels)
    for X, y in train_iter:
        print(net(X.cuda())[:1].reshape(-1,8,3))
        print(y[:1].reshape(-1,8,3))
        break
    # for _ in range(num_epochs):
    #     train_iter = data_iter(batch_size, features, labels)
    #     train(net, train_iter, loss_func, trainer)
    # torch.save(net.state_dict(), PATH)
    


