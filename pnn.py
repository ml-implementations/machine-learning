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
    def __init__(self, mode='train_cifar10_pnn'):
        self.mode = mode
        self.cifar10_train_loader = None
        self.cifar10_test_loader = None
        self.cifar10_net = None
        self.cifar10_epochs = 50
        self.cifar10_path = 'models/cifar10'

        self.cifar10_criterion = nn.CrossEntropyLoss()
        self.cifar10_optimizer = None

        self.mnist_net = None
        self.mnist_epochs = 50
        self.mnist_train_loader = None
        self.mnist_test_loader = None
        self.mnist_path = 'models/mnist'

        self.mnist_criterion = nn.CrossEntropyLoss()
        self.mnist_optimizer = None

        self.cifar10_net_pnn = None
        self.cifar10_epochs_pnn = 50
        self.cifar10_path_pnn = 'models/cifar10_pnn'

        self.cifar10_criterion_pnn = nn.CrossEntropyLoss()
        self.cifar10_optimizer_pnn = None

    def load_cifar10_dataset(self):
        transform = transforms.Compose(
            [#transforms.Grayscale(),
             transforms.ToTensor()]
        )

        train_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
        self.cifar10_train_loader = torch.utils.data.DataLoader(train_set, batch_size=8,
                                                                shuffle=True, num_workers=0)

        test_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
        self.cifar10_test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                                               shuffle=False, num_workers=0)

    def load_fashion_mnist_dataset(self):
        transform = transforms.Compose(
            [transforms.ToTensor()])

        train_set = torchvision.datasets.FashionMNIST(root='./data/fashion-mnist', train=True, download=True,
                                                      transform=transform)
        self.mnist_train_loader = torch.utils.data.DataLoader(train_set, batch_size=8,
                                                              shuffle=True, num_workers=0)

        test_set = torchvision.datasets.FashionMNIST(root='./data/fashion-mnist', train=False, download=True,
                                                     transform=transform)
        self.mnist_test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                                             shuffle=False, num_workers=0)

    def train_cifar10(self):

        for epoch in range(self.cifar10_epochs):

            running_loss = 0.0
            for i, data in enumerate(self.cifar10_train_loader, 0):
                inputs, labels = data

                # zero the parameter gradients
                self.cifar10_optimizer.zero_grad()

                # forward + backward + optimize
                # inputs = torch.flatten(torch.tensor(inputs, device='cuda'), start_dim=1)
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
                # inputs = torch.flatten(torch.tensor(inputs, device='cuda'), start_dim=1)
                outputs, outputs_layers = self.mnist_net.forward(torch.tensor(inputs, device='cuda'))
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

    def train_cifar10_pnn(self):
        mnist_train_loader_iter = iter(self.mnist_train_loader)

        for epoch in range(self.cifar10_epochs_pnn):

            running_loss = 0.0
            for i, data in enumerate(self.cifar10_train_loader, 0):

                try:
                    mnist_data = next(mnist_train_loader_iter)
                except StopIteration:
                    mnist_train_loader_iter = iter(self.mnist_train_loader)
                    mnist_data = next(mnist_train_loader_iter)

                inputs, labels = data
                inputs_mnist, labels_mnist = mnist_data

                # zero the parameter gradients
                self.cifar10_optimizer_pnn.zero_grad()

                # forward + backward + optimize
                #inputs = torch.flatten(torch.tensor(inputs, device='cuda'), start_dim=1)

                #inputs_mnist = torch.flatten(torch.tensor(inputs_mnist, device='cuda'), start_dim=1)
                outputs_mnist, outputs_layers_mnist = self.mnist_net.forward(torch.tensor(inputs_mnist, device='cuda'))
                outputs = self.cifar10_net_pnn.forward(torch.tensor(inputs, device='cuda'),
                                                       outputs_layers_mnist['conv1'],
                                                       outputs_layers_mnist['fc2'],
                                                       outputs_layers_mnist['fc3'],
                                                       outputs_layers_mnist['fc4'])
                loss = self.cifar10_criterion_pnn(outputs, labels)
                loss.backward()
                self.cifar10_optimizer_pnn.step()

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
                images = torch.flatten(torch.tensor(images, device='cuda'), start_dim=1)
                outputs = self.cifar10_net.forward(images)
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
                images = torch.flatten(torch.tensor(images, device='cuda'), start_dim=1)
                outputs, outputs_layers = self.mnist_net.forward(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (
                100 * correct / total))

    def test_cifar10_pnn(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.cifar10_test_loader:
                images, labels = data
                images = torch.flatten(torch.tensor(images, device='cuda'), start_dim=1)
                outputs = self.cifar10_net_pnn.forward(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))

    def load_cifar10(self):
        if self.mode == 'train_cifar10':
            self.cifar10_net = Cifar10Net()
        elif self.mode == 'test_cifar10':
            self.cifar10_net = Cifar10Net()
            self.cifar10_net.load_state_dict(torch.load(self.cifar10_path))
            self.cifar10_net.eval()

    def load_mnist(self):
        if self.mode == 'train_mnist':
            self.mnist_net = FashionMNISTNet()
        elif self.mode == 'test_mnist' or self.mode == 'train_cifar10_pnn':
            self.mnist_net = FashionMNISTNet()
            self.mnist_net.load_state_dict(torch.load(self.mnist_path))
            self.mnist_net.eval()
            for p in self.mnist_net.parameters():
                p.requires_grad = False

    def load_cifar10_pnn(self):
        if self.mode == 'train_cifar10_pnn':
            self.cifar10_net_pnn = Cifar10NetPNN(3, 64, 32, 10)
        elif self.mode == 'test_cifar10_pnn':
            self.cifar10_net_pnn = Cifar10NetPNN(3, 64, 32, 10)
            self.cifar10_net_pnn.load_state_dict(torch.load(self.cifar10_path_pnn))
            self.cifar10_net_pnn.eval()

    def save_cifar10(self):
        torch.save(self.cifar10_net.state_dict(), self.cifar10_path)

    def save_mnist(self):
        torch.save(self.mnist_net.state_dict(), self.mnist_path)

    def save_cifar10_pnn(self):
        torch.save(self.cifar10_net_pnn.state_dict(), self.cifar10_path_pnn)

    def train(self):
        self.load_cifar10_dataset()
        self.load_fashion_mnist_dataset()
        self.load_mnist()
        self.load_cifar10()
        self.load_cifar10_pnn()
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
        elif self.mode == 'test_mnist':
            self.test_mnist()
        elif self.mode == 'train_cifar10_pnn':
            self.cifar10_optimizer_pnn = optim.SGD(self.cifar10_net_pnn.parameters(), lr=0.001, momentum=0.9)
            self.train_cifar10_pnn()
            self.save_cifar10_pnn()
        elif self.mode == 'test_cifar10_pnn':
            self.test_cifar10_pnn()


class Cifar10Net(nn.Module):
    def __init__(self):
        super(Cifar10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 8, 5, stride=2)
        self.fc3 = nn.Linear(12 * 12 * 8, 120)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.softmax(self.fc5(x), dim=1)
        return x


class FashionMNISTNet(nn.Module):
    def __init__(self):
        super(FashionMNISTNet, self).__init__()

        # Defining the layers, 128, 64, 10 units each
        self.conv1 = nn.Conv2d(1, 3, 5, stride=2)
        self.fc2 = nn.Linear(12 * 12 * 3, 64)
        self.fc3 = nn.Linear(64, 32)

        # Output layer, 10 units - one for each digit
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        out = dict()
        out['conv1'] = F.relu(self.conv1(x))
        out['fc2'] = F.relu(self.fc2(torch.flatten(out['conv1'], start_dim=1)))
        out['fc3'] = F.relu(self.fc3(out['fc2']))
        out['fc4'] = F.softmax(self.fc4(out['fc3']), dim=1)

        return out['fc4'], out


class Cifar10NetPNN(nn.Module):
    def __init__(self, conv1_in, fc2_in, fc3_in, fc4_in):
        super(Cifar10NetPNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 8, 5, stride=2)
        self.fc3 = nn.Linear(12 * 12 * 8, 120)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, 10)

        self.conv1_mnist = nn.Conv2d(conv1_in, 8, 3, padding=1)
        self.fc2_mnist = nn.Linear(fc2_in, 120) # fc2_in = 588
        self.fc3_mnist = nn.Linear(fc3_in, 84)
        self.fc4_mnist = nn.Linear(fc4_in, 10)

    def forward(self, x, conv1_mnist_out, fc2_mnist_out, fc3_mnist_out, fc4_mnist_out):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x) + self.conv1_mnist(conv1_mnist_out))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc3(x) + self.fc2_mnist(fc2_mnist_out))
        x = F.relu(self.fc4(x) + self.fc3_mnist(fc3_mnist_out))
        x = F.softmax(self.fc5(x), dim=1)
        return x


def main():
    fire.Fire(PNN)


if __name__ == '__main__':
    main()
