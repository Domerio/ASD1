import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.optim import lr_scheduler

# cuda helps in utilizing the GPU for faster computation
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Imagesize = (64, 64)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(Imagesize),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                ])

trainset = torchvision.datasets.ImageFolder(root='data/train', transform=transform)
testset = torchvision.datasets.ImageFolder(root='data/valid', transform=transform)

print("dataset has the following classes ", trainset.classes)
print(type(trainset))
num_classes = len(trainset.classes)
print(num_classes)

trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=32, shuffle=False)

resnet50 = torchvision.models.resnet50(pretrained=True)

fr = resnet50.fc.in_features
resnet50.fc = nn.Linear(fr, 2)
use_cuda = torch.cuda.is_available()

# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet50.parameters(), lr=0.001)

# 在创建optimizer之后，可以设置一个学习率调度器，例如StepLR
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

TrainLoss = []
TrainAcc = []
TestLoss = []
TestAcc = []
num_epochs = 20

best_accuracy = 0.0


def train(TrainLoss, TrainAcc, TestLoss, TestAcc, num_epochs, model, trainloader, testloader, criterion, optimizer,
          scheduler=None):
    global best_accuracy

    total_step = len(trainloader)

    for epoch in range(num_epochs):
        trainAcc = 0
        testAcc = 0

        # Training
        model.train()
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            trainLoss = criterion(outputs, labels)
            trainLoss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            trainAcc += torch.sum(preds == labels)

        trainAcc = trainAcc.float() / len(trainloader.dataset) * 100

        # Testing
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(testloader):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                testLoss = criterion(outputs, labels)

                preds = outputs.argmax(dim=1)
                testAcc += torch.sum(preds == labels)

            testAcc = testAcc.float() / len(testloader.dataset) * 100

        # Print and save results
        print(
            "Epoch {} =>  loss : {:.2f};   Accuracy : {:.2f}%;   test_loss : {:.2f};   test_Accuracy : {:.2f}%".format(
                epoch + 1, trainLoss.item(), trainAcc, testLoss.item(), testAcc))

        TrainLoss.append(trainLoss)
        TrainAcc.append(trainAcc)
        TestLoss.append(testLoss)
        TestAcc.append(testAcc)

        # Save best model
        if testAcc > best_accuracy:
            best_accuracy = testAcc
            torch.save(model.state_dict(), 'resnet50_1.pth')
            print("Best model saved with accuracy: {}%".format(best_accuracy))

        # Update learning rate using scheduler
        if scheduler is not None:
            scheduler.step()


train(TrainLoss, TrainAcc, TestLoss, TestAcc, num_epochs, resnet50, trainloader, testloader, criterion, optimizer)
