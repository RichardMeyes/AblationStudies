import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable


def plot_data():

    def imshow(img):
        img = img / 2 + 0.5  # un-normalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # plot some data
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 20, bias=False)
        self.fc2 = nn.Linear(20, 10, bias=False)
        self.fc3 = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)  # needs NLLLos() loss
        return x

    def train_net(self, criterion, optimizer, trainloader, epochs, device):
        # save untrained net
        if save:
            torch.save(net.state_dict(), '../nets/MNIST_MLP(20, 10)_untrained.pt')
        # train the net
        log_interval = 10
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = Variable(data), Variable(target)
                data, target = data.to(device), target.to(device)
                # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
                data = data.view(-1, 28 * 28)
                optimizer.zero_grad()
                net_out = self(data)
                loss = criterion(net_out, target)
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                                   len(trainloader.dataset),
                                                                                   100. * batch_idx / len(trainloader),
                                                                                   loss.data.item()))
        if save:
            # save trained net
            torch.save(net.state_dict(), '../nets/MNIST_MLP(20, 10)_trained.pt')

    def test_net(self, criterion, testloader, device):
        # test the net
        test_loss = 0
        correct = 0
        correct_class = np.zeros(10)
        correct_labels = np.array([], dtype=int)
        class_labels = np.array([], dtype=int)
        for i_batch, (data, target) in enumerate(testloader):
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28 * 28)
            net_out = self(data)
            # sum up batch loss
            test_loss += criterion(net_out, target).data.item()
            pred = net_out.data.max(1)[1]  # get the index of the max log-probability
            batch_labels = pred.eq(target.data)
            correct_labels = np.append(correct_labels, batch_labels)
            class_labels = np.append(class_labels, target.data)
            for i_label in range(len(target)):
                label = target[i_label].item()
                correct_class[label] += batch_labels[i_label].item()
            correct += batch_labels.sum()
        test_loss /= len(testloader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct,
                                                                                     len(testloader.dataset),
                                                                                     100. * correct.item() / len(
                                                                                         testloader.dataset)))
        acc = 100. * correct.item() / len(testloader.dataset)
        # calculate class_acc
        acc_class = np.zeros(10)
        for i_label in range(10):
            num = (testloader.dataset.test_labels.numpy() == i_label).sum()
            acc_class[i_label] = correct_class[i_label]/num
        return acc, correct_labels, acc_class, class_labels


if __name__ == "__main__":

    """ setting flags """
    # chose data plotting
    plot = False
    # chose CPU or GPU:
    dev = "GPU"
    # chose training or loading pre-trained model
    train = True
    save = False
    test = True

    # prepare GPU
    if dev == "GPU":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu:0")
    print("current device:", device)

    # build net
    net = Net()
    if dev == "GPU":
        print("sending net to GPU")
        net.to(device)
    criterion = nn.NLLLoss()  # nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # load data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    if plot:
        plot_data()

    if train:
        net.train_net(criterion, optimizer, trainloader, epochs=10, device=device)
    else:
        net.load_state_dict(torch.load('../nets/MNIST_MLP(20, 10)_trained.pt'))
        net.eval()

    if test:
        acc, correct_labels, acc_class = net.test_net(criterion, testloader, device)
        print(acc)
        print(correct_labels)
        print(acc_class)
        print(acc_class.mean())  # NOTE: This does not equal to the calculated total accuracy as the distribution of labels is not equal in the test set!