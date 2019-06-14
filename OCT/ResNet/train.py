from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from base_dataset import Preproc, Rescale, RandomCrop, ToTensor, Normalization, Resize

from tqdm import tqdm

if __name__ == '__main__':

    plt.ion()   # interactive mode

    data_dir = '../datas/'
    BATCH_SIZE = 8

    data_transforms = {
        'train_data': transforms.Compose([   
            Preproc(0.2),                # 裁剪白边
            Rescale(224),                # 重新调整图片分辨率
            transforms.CenterCrop(224),  # 将图片中心切割成为指定大小的正方形图片
            ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    #预训练模型已经训练好的参数
        ]),
        'val_data': transforms.Compose([
            Preproc(0.2),
            Rescale(224),
            transforms.CenterCrop(224),
            ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                    for x in ['train_data', 'val_data']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=4)
                for x in ['train_data', 'val_data']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train_data', 'val_data']}
    class_names = image_datasets['train_data'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated


    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train_data']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    # imshow(out, title=[class_names[x] for x in classes])


    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())     # 深拷贝当前模型的参数
        best_loss = 999999.9

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train_data', 'val_data']:
                if phase == 'train_data':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                step = 0
                for inputs, labels in dataloaders[phase]:
                    step += 1
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train_data'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train_data':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data)

                    if phase == 'train_data' and step%32==0:
                        print('Epoch: {} {}/{} Loss: {:.4f} Acc: {:.4f}'.format(epoch, step*inputs.size(0), dataset_sizes[phase], running_loss/(step*inputs.size(0)), running_corrects.float()/(step*inputs.size(0))))

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val_data' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), 'best_model.pth')
                    print ('Best Epoch: {}'.format(epoch))

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders['val_data']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('label: {} / pred: {}'.format(class_names[labels[j]], class_names[preds[j]]))
                    imshow(inputs.cpu().data[j])
                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

    model_ft = models.resnet34(pretrained=True)

    lt = 8
    cntr = 0
    for child in model_ft.children():
        cntr += 1
        if cntr < lt:
            print(child)
            for param in child.parameters():
                param.requires_grad = False        # 冻结前27层

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 4)        # 分为4类

    model_ft = model_ft.to(device)             # 在设备上加载模型

    criterion = nn.CrossEntropyLoss()          # 创建一个损失函数对象

    # Adam 是一种可以替代传统随机梯度下降过程的一阶优化算法，它能基于训练数据迭代地更新神经网络权重，学习率设为0.001。
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.001)

    # Decay LR by a factor of 0.1 every epoch
    # 每一周期（epoch）将学习率乘以0.1
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=6)

    visualize_model(model_ft)
    plt.ioff()
    plt.show()
