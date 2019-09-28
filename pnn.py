import torch.nn as nn
import torch
import fire
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
from multiprocessing import set_start_method

torch.set_default_tensor_type(torch.cuda.FloatTensor)

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class PNN:
    def __init__(self):
        self.mode = 'train_mnist'
        self.cifar10_train_loader = None
        self.cifar10_test_loader = None
        self.cifar10_net = None
        self.cifar10_epochs = 50
        self.cifar10_path = 'models/cifar10'

        self.cifar10_criterion = nn.CrossEntropyLoss()
        self.cifar10_optimizer = None

        self.mnist_net = None
        self.mnist_epochs = 5
        self.mnist_train_loader = None
        self.mnist_test_loader = None
        self.mnist_path = 'models/mnist'

        self.mnist_criterion = nn.CrossEntropyLoss()
        self.mnist_optimizer = None

    def load_cifar10_dataset(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=False, transform=transform)
        self.cifar10_train_loader = torch.utils.data.DataLoader(train_set, batch_size=8,
                                                                shuffle=True, num_workers=1)

        test_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=False, transform=transform)
        self.cifar10_test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                                               shuffle=False, num_workers=1)

    def load_fashion_mnist_dataset(self):
        transform = transforms.Compose(
            [transforms.ToTensor()])

        train_set = torchvision.datasets.FashionMNIST(root='./data/fashion-mnist', train=True, download=False,
                                                      transform=transform)
        self.mnist_train_loader = torch.utils.data.DataLoader(train_set, batch_size=8,
                                                              shuffle=True, num_workers=1)

        test_set = torchvision.datasets.FashionMNIST(root='./data/fashion-mnist', train=False, download=False,
                                                     transform=transform)
        self.mnist_test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                                             shuffle=False, num_workers=1)

    def train_cifar10(self):

        for epoch in range(self.cifar10_epochs):

            running_loss = 0.0
            for i, data in enumerate(self.cifar10_train_loader, 0):
                inputs, labels = data

                # zero the parameter gradients
                self.cifar10_optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.cifar10_net.forward(torch.tensor(inputs, device='cuda'))
                loss = self.cifar10_criterion(outputs, labels)
                loss.backward()
                self.cifar10_optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

    def train_mnist(self):
        for epoch in range(self.mnist_epochs):

            running_loss = 0.0
            for i, data in enumerate(self.mnist_train_loader, 0):
                inputs, labels = data

                # zero the parameter gradients
                self.mnist_optimizer.zero_grad()

                # forward + backward + optimize
                inputs = torch.flatten(torch.tensor(inputs, device='cuda'), start_dim=1)
                outputs = self.mnist_net.forward(torch.tensor(inputs, device='cuda'))
                loss = self.mnist_criterion(outputs, labels)
                loss.backward()
                self.mnist_optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

    def test_cifar10(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.cifar10_test_loader:
                images, labels = data
                outputs = self.cifar10_net.forward(torch.tensor(images, device='cuda'))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))

    def test_mnist(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.mnist_test_loader:
                images, labels = data
                outputs = self.mnist_net.forward(torch.tensor(images, device='cuda'))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (
                100 * correct / total))

    def load_cifar10(self):
        if self.mode == 'train_cifar10':
            self.cifar10_net = Cifar10Net()
        else:
            self.cifar10_net = Cifar10Net()
            self.cifar10_net.load_state_dict(torch.load(self.cifar10_path))
            self.cifar10_net.eval()

    def load_mnist(self):
        self.mnist_net = FashionMNISTNet()

    def save_cifar10(self):
        torch.save(self.cifar10_net.state_dict(), self.cifar10_path)

    def save_mnist(self):
        torch.save(self.mnist_net.state_dict(), self.mnist_path)

    def train(self):
        self.load_cifar10_dataset()
        self.load_fashion_mnist_dataset()
        self.load_mnist()
        self.load_cifar10()
        if self.mode == 'train_cifar10':
            self.cifar10_optimizer = optim.SGD(self.cifar10_net.parameters(), lr=0.001, momentum=0.9)
            self.train_cifar10()
            self.save_cifar10()
        elif self.mode == 'test_cifar10':
            self.test_cifar10()
        elif self.mode == 'train_mnist':
            self.mnist_optimizer = optim.SGD(self.mnist_net.parameters(), lr=0.001, momentum=0.9)
            self.train_mnist()
            self.save_mnist()


class Cifar10Net(nn.Module):
    def __init__(self):
        super(Cifar10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FashionMNISTNet(nn.Module):
    def __init__(self):
        super(FashionMNISTNet, self).__init__()

        # Defining the layers, 128, 64, 10 units each
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)

        # Output layer, 10 units - one for each digit
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.softmax(x, dim=1)

        return x


if __name__ == '__main__':
    fire.Fire(PNN)
