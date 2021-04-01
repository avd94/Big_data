import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from utils import _get_model_file
import os
import time

def save_model(net, model_path, model_name):
    path = os.path.join(model_path, 'model')
    os.makedirs(path, exist_ok=True)
    model_file = _get_model_file(path, model_name)
    torch.save(net.state_dict(), model_file)

def custom_train(net, trainloader, valloader, testloader, dir, model_num, epochs=1, lr=0.001, device=None):
    ###################### Define Loss function and optimizer
    lossfunc = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, threshold=0.0001)
    train_losses = list()
    train_scores = list()
    test_scores = list()
    time_list = list()
    net.to(device)
    experiment_timer = time.time()
    for epoch in range(epochs):
        print('Running epoch {}'.format(epoch))
        print('-' * 10)
        net.train()
        epoch_loss = 0
        true_preds, count = 0., 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net.forward(inputs)
            loss = lossfunc(outputs, labels)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            true_preds += (outputs.argmax(dim=-1) == labels).sum().item()
            count += labels.shape[0]
        epoch_loss = epoch_loss / (i + 1)
        train_losses.append(epoch_loss)
        train_acc = true_preds / count
        train_scores.append(train_acc)
        print("[Epoch %2i] Training loss: %05.2f, Training accuracy: %05.2f%%" %(
            epoch + 1, epoch_loss, train_acc * 100.0))
        val_val = val_loss(net, valloader, device=device)
        scheduler.step(val_val)
        test_acc = valid(net, testloader, device=device)
        test_scores.append(test_acc['accuracy'])
        epoch_time = time.time() - experiment_timer
        time_list.append(epoch_time)

    results = {"train_losses": train_losses,
               "train_scores": train_scores,
               "test_scores": test_scores,
               "training_times": time_list}
    save_model(net, dir, model_num)
    return results

def valid(net, testloader, device = None):
    correct = 0
    total = 0
    with torch.no_grad():
         for data in testloader:
             images, labels = data
             if device is not None:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    dict_result = {'accuracy': 100*correct/total}
    return dict_result

def val_loss(net, loader, device=None):
    lossfunc = nn.CrossEntropyLoss()
    epoch_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(loader):
            images, labels = data
            if device is not None:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = lossfunc(outputs, labels)
                epoch_loss += loss.item()
    return epoch_loss/(i+1)



def valid_class(net, testloader, classes, device = None):
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    dict_result = dict()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if device is not None:
              images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        dict_result['accuracy_class_{}'.format(i)] = 100 * class_correct[i] / class_total[i]

    return dict_result


