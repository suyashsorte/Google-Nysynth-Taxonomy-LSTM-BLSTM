import torch
import os
import torchvision
import torchvision.datasets as dataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.utils.data as data
from random import shuffle
from pytorch_nsynth.nsynth import NSynth
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sklearn.metrics as sk
import wave
import time as t
import sys

import pandas as pd


def get_free_gpu():
    # Invoke a system call to nvidia-smi, and filter it using the unix 'grep command.'
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp-gpu')

    # Parse out the available memory for each available device.
    memory_available = [int(x.split()[2]) for x in open('tmp-gpu', 'r').readlines()]
    for index in range(0, len(memory_available)):
        print("GPU " + str(index) + "\t " + str(memory_available[index]))
    return np.argmax(memory_available)


device = torch.device('cuda:' + str(get_free_gpu()) if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose(
    [transforms.Lambda(lambda x: x / np.iinfo(np.int16).max),
     transforms.Lambda(lambda x: torch.from_numpy(x).float()), transforms.Lambda(lambda x: x[0:16000])])

# Loading train data
train_dataset = NSynth(
    "/local/sandbox/nsynth/nsynth-train",
    transform=transform,
    blacklist_pattern=["synth_lead"],  # blacklist string istrument
    categorical_field_list=["instrument_family", "instrument_source"])

train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)

# Loading test data
test_dataset = NSynth(
    "/local/sandbox/nsynth/nsynth-test",
    transform=transform,
    blacklist_pattern=["synth_lead"],  # blacklist string instrument
    categorical_field_list=["instrument_family", "instrument_source"])
test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=True)
loss_validation = []
loss_train = []

# Loading validation data
valid_dataset = NSynth(
    "/local/sandbox/nsynth/nsynth-valid",
    transform=transform,
    blacklist_pattern=["synth_lead"],  # blacklist string instrument
    categorical_field_list=["instrument_family", "instrument_source"])

valid_loader = data.DataLoader(valid_dataset, batch_size=128, shuffle=True)

plt.figure()
# Visualising 1D audio waveform
classes = ('bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'vocal')
for i in range(10):

    plt.plot((test_dataset[i][0]).cpu().numpy())
    # plt.show()
    name = "Class " + classes[i] + ".png"
    print("saving 1D audio waveform of " + "Class " + classes[i])
    plt.savefig(name)
    plt.close()


class RNN(nn.Module):
    def __init__(self, input_size=100, hidden_size=128, num_layers=2, num_classes=10):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out





# net = Net()
net = RNN()
#net.load_state_dict(torch.load("dict_model.pwf"))
net.to(device)
# net = RNN()
# net.to(device)
target = []


def train_data(train_loader):
    net.train()

    for sample, instrument_family, instrument_source_target, targets in train_loader:
        sample, instrument_family = sample.reshape(-1, 160, 100).to(device), instrument_family.to(device)
        # sample = sample.unsqueeze(1)  # -----------------------
        # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.8)
        optimizer = optim.Adam(net.parameters(), lr=0.0005)
        optimizer.zero_grad()
        output = net(sample)
        loss = F.cross_entropy(output, instrument_family)
        loss.backward()
        optimizer.step()
    loss_train.append(loss)


pred = []


def validation():
    net.eval()
    val_loss = 0
    correct = 0
    val_len = len(valid_loader.dataset)
    output = 0
    for sample, instrument_family, instrument_source_target, targets in valid_loader:
        sample, instrument_family = sample.reshape(-1, 160, 100).to(device), instrument_family.to(device)
        # sample = sample.unsqueeze(1)
        output = net(sample)
        val_loss += F.cross_entropy(output, instrument_family, size_average=False).data[0]
        predict = output.data.max(1, keepdim=True)[1]
        correct += predict.eq(instrument_family.data.view_as(predict)).cpu().sum()
        target.append(instrument_family)
        pred.append(predict)

    val_loss /= val_len
    loss_validation.append(val_loss)
    print('Validation Average loss: {:.4f}'.format(val_loss))
    print("Accuracy: {}/{} ({:.2f}%)".format(correct, val_len, 100. * correct / val_len))


confusion_matrix = [[0. for i in range(10)] for j in range(10)]
class_accuracy = {}
one_d = [0. for i in range(10)]
two_d = [0. for i in range(10)]
list_of_index = [0. for i in range(10)]
list_of_index2 = [0. for i in range(10)]


def test():
    net.eval()
    val_loss = 0
    correct = 0
    val_len = len(test_loader.dataset)
    for sample, instrument_family, instrument_source_target, targets in test_loader:

        sample, instrument_family = sample.reshape(-1, 160, 100).to(device), instrument_family.to(device)
        # sample = sample.unsqueeze(1)
        output = net(sample)
        val_loss += F.cross_entropy(output, instrument_family, size_average=False).data[0]
        predict = output.data.max(1, keepdim=True)[1]
        correct += predict.eq(instrument_family.data.view_as(predict)).cpu().sum()

        np_output = (output).cpu()
        ar_output = (output).data.cpu().numpy()
        sorted = []
        for i in range(len(np_output)):
            maximum, index = torch.max(np_output[i], 0)
            np_output[i][index] = -100000000
            second_max, index2 = torch.max(np_output[i], 0)

            a = instrument_family[i].cpu()
            if a == index:
                one_d[a] = sample[index][0].data.cpu().numpy()
                list_of_index[a] = index.data
                two_d[a] = sample[index2][0].data.cpu().numpy()
                list_of_index2[a] = index2.data

        for i in range(len(instrument_family)):
            confusion_matrix[instrument_family[i]][predict[i]] += 1
        target.append(instrument_family)
        pred.append(predict)
    val_loss /= val_len
    # loss_validation.append(val_loss)

    print('Test Average loss: {:.4f}'.format(val_loss))
    print("Accuracy: {}/{} ({:.2f}%)".format(correct, val_len, 100. * correct / val_len))
    if i not in class_accuracy:
        class_accuracy[i] = 100. * correct / val_len
    else:
        class_accuracy[i] += 100. * correct / val_len


def learning_curve():
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(0, len(loss_validation)))
    plt.plot(loss_train, label="Train")
    plt.plot(loss_validation, label="Validation")

    plt.legend()
    # plt.show()
    plt.savefig("Learning Curve.png")
    plt.close()


if __name__ == '__main__':
    epoch = 10
    start = t.time()
    for e in range(epoch):
        # break
        print(e + 1)
        # calling train function
        train_data(train_loader)
        print("doing validation")
        # calling validation
        validation()
    print("Total time for training:", t.time() - start)
    # uncomment below line to save weights of the model
    # torch.save(net.state_dict(), "dict_model.pwf")
    # testing the model
    print("Accuracy of Test")
    test()
    # calling learning curve and saving it
    learning_curve()
    pitch = []
    exit()
    # displaying confusion matrix
    print("Confusion Matrix")
    for i in confusion_matrix:
        print(i)

    plt.imshow(confusion_matrix, cmap='hot', interpolation='nearest')
    plt.savefig("confusion matrix.png")
    plt.close()
    # getting class accuracies of each class
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for sample, instrument_family, instrument_source_target, targets in test_loader:
        sample, instrument_family = sample.reshape(-1, 160, 100).to(device), instrument_family.to(device)
        # sample = sample.unsqueeze(1)

        outputs = net(sample)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == instrument_family).squeeze()
        for i in range(4):
            label = instrument_family[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
    print("Accuracy of each class")
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    j = 0
    # Displaying waveform of correct claass and near correct class
    for i, k in zip(one_d, two_d):
        print("for correct class ", classes[list_of_index[j]], "tensor:", list_of_index[j])
        plt.plot(i)
        # plt.show()
        name2 = "for correct class " + classes[list_of_index[j]] + ".png"
        plt.savefig(name2)
        plt.close()

        print("for near correct class ", classes[list_of_index2[j]], "tensor:", list_of_index2[j])
        plt.plot(k)
        # plt.show()
        name1 = "for near correct class " + classes[list_of_index2[j]] + " for Correct " + classes[
            list_of_index[j]] + ".png"
        plt.savefig(name1)
        plt.close()

        j = j + 1
