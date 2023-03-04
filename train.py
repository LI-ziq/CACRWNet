import torch
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from ResNet50_Deeplabv3 import *
from ResNet50 import *
from resnet18 import *
import shutil
import config
from spliy import spliy



#数据增强
# data_transform = {
#     # 数据预处理
#     "train": transforms.Compose(
#         [
#         transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),  # 随机裁剪到156*156
#         transforms.RandomRotation(degrees=45),  # 随机旋转
#         transforms.RandomHorizontalFlip(),  # 随机水平翻转
#         transforms.CenterCrop(size=224),  # 中心裁剪到124*124
#         transforms.ToTensor(),  # 转化成张量
#         transforms.Normalize([0.485, 0.456, 0.406],  # 归一化
#                              [0.229, 0.224, 0.225])
#          ]),
#
#     "val": transforms.Compose(
#         [transforms.Resize(256),
#          transforms.CenterCrop(224),
#          transforms.ToTensor(),
#          transforms.Normalize([0.485, 0.456, 0.406],
#                               [0.229, 0.224, 0.225])
#                                ])}

data_transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#利用Dataloader加载数据
train_directory = config.TRAIN_DATASET_DIR
valid_directory = config.VALID_DATASET_DIR

batch_size = config.BATCH_SIZE
num_classes = config.NUM_CLASSES

train_datasets = datasets.ImageFolder(('./UCMdataset/train'),
                                     transform=data_transform["train"])
# 加载验证集
valid_datasets = datasets.ImageFolder(('./UCMdataset/val'),
                                        transform=data_transform["val"])

train_data_size = len(train_datasets)
train_data = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

valid_data_size = len(valid_datasets)
valid_data = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)

# print(train_data_size, valid_data_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #若有gpu可用则用gpu
#使用Resnet-50的预训练模型进行迁移学习
# model = resnet50_DV()
# model = resnet50()
model = resnet18_DV()

model.to(device)
# testmodel = model
#查看更改后的模型参数
#print('after:{%s}\n'%resnet50)

#定义损失函数和优化器
# loss_func = nn.NLLLoss()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=3e-4, amsgrad=True, weight_decay=0.0001)

#训练过程
def train_and_valid(model, loss_function, optimizer, epochs, e):


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #若有gpu可用则用gpu
    # scaler = torch.cuda.amp.GradScaler()
    # autocast = torch.cuda.amp.autocast

    # print("using {} device.".format(device))
    # print("PyTorch Version: ", torch.__version__)
    print("using {} images for training, {} images for validation.".format(
        train_data_size, valid_data_size))
    params = sum([v.numel() for k, v in model.state_dict().items()])
    # 打印信息
    print(params)
    record = []
    best_acc = 0.0
    Bestacc = []
    best_epoch = 0
    writer = SummaryWriter('logs')
    writer_acc = SummaryWriter('acc')

    for epoch in range(epochs): #训练epochs轮
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train() #训练

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        train_bar = tqdm(train_data, file=sys.stdout)
        for i, (inputs, labels) in enumerate(train_bar):

            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(labels)
            optimizer.zero_grad()

            # with autocast():
            optimizer.zero_grad() #梯度清零

            outputs = model(inputs) #数据前馈，正向传播

            loss = loss_function(outputs, labels) #输出误差

            # scaler.scale(loss).backward()
            #
            # scaler.step(optimizer)
            #
            # scaler.update()
            #
            loss.backward() #反向传播

            optimizer.step() #优化器更新参数

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)
            train_bar.desc = "train epoch[{}/{}]——loss:{:.3f}".format(epoch + 1,
                                                                      epochs,
                                                                      loss)

        with torch.no_grad():
            model.eval() #验证
            val_bar = tqdm(valid_data, file=sys.stdout)
            for j, (inputs, labels) in enumerate(val_bar):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = loss_function(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)
                val_bar.desc = "valid epoch——[{}/{}]".format(epoch + 1,
                                                             epochs)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        writer.add_scalar('train', avg_train_loss, global_step=epoch)
        writer_acc.add_scalar('acc', avg_valid_acc, global_step=epoch)

        record.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if avg_valid_acc > best_acc  : #记录最高准确性的模型
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            torch.save(model, f'{e+1}_model.pth')
            # torch.save(model.state_dict(), 'resnet50meiyoue_model_onlyweigths.pth')
            torch.save(model.state_dict(), 'resnet18_new_all.pth')
        # if  avg_train_acc > 98.4:  # 记录最高准确性的模型
        #     best_acc = avg_train_acc
        #     best_epoch = epoch + 1
        #     torch.save(model, f'{e + 1}_model.pth')


        epoch_end = time.time()

        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
    Bestacc = best_acc

        # torch.save(model, 'trained_models/resnet50_model_' + str(epoch + 1) + '.pth')
    return model, record, Bestacc



#结果
if __name__=='__main__':
    num_epochs = config.NUM_EPOCHS
    num = 1
    # Best = 0
    # list = []
    for epoch in range(num):  # 训练epochs轮
        print("Experiment: {}/{}".format(epoch + 1, num))
        trained_model, record, Best = train_and_valid(model, loss_func, optimizer, num_epochs, epoch)
        # list.append(Best)
        # print('delete dataset')
        # shutil.rmtree('./dataset')
        # print('creat dataset')
        # spliy()
    # print(list)
    # print('mean', np.mean(list))
        torch.save(record, config.TRAINED_MODEL)

    # record = np.array(record)
    # plt.plot(record[:, 0:2])
    # plt.legend(['Train Loss', 'Valid Loss'])
    # plt.xlabel('Epoch Number')
    # plt.ylabel('Loss')
    # plt.ylim(0, 1)
    # plt.savefig('loss.png')
    # plt.show()
    #
    # plt.plot(record[:, 2:4])
    # plt.legend(['Train Accuracy', 'Valid Accuracy'])
    # plt.xlabel('Epoch Number')
    # plt.ylabel('Accuracy')
    # plt.ylim(0, 1)
    # plt.savefig('accuracy.png')
    # plt.show()
#
# '''
#     model = torch.load('trained_models/resnet50_model_23.pth')
#     predict(model, '61.png')
# '''