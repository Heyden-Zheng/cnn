import torch
import torchvision
import torch.utils.data as Data
import torch.nn as nn
from model import CNN

# 定义超参数
from torch.autograd import Variable

EPOCH = 3
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False  # 已经有了，无需重复下载

train_data = torchvision.datasets.MNIST(root='./data',
                                        train=True,
                                        # transform是将原数据规范化到0-1区间
                                        transform=torchvision.transforms.ToTensor(),
                                        download=DOWNLOAD_MNIST)

test_data = torchvision.datasets.MNIST(root='./data',
                                       train=False,
                                       transform=torchvision.transforms.ToTensor())

# 使用DataLoader加载数据集
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=BATCH_SIZE,
                               shuffle=True)
test_loader = Data.DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

test_x = Variable(torch.unsqueeze(test_data.data, dim=1)).type(torch.FloatTensor)
test_y = test_data.targets

cnn = CNN()
# 损失函数
loss_function = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)


# 训练模型
def train():
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(y)

            output = cnn(b_x)
            loss = loss_function(output, b_y)
            optimizer.zero_grad()  # 反向传播前清空梯度

            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = sum(pred_y == test_y) / test_y.size(0)
                print('Epoch:', epoch+1, 'Step:', step, '|train loss:%.4f' % loss.item(),
                      'test accuracy:%.4f' % accuracy)


def test():
    test_output = cnn(test_x[:20])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y[:20], 'prediction number')
    print(test_y[:20].numpy(), 'real number')


if __name__ == '__main__':
    train()
    test()
