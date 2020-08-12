import fire
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


torch.set_default_tensor_type(torch.cuda.FloatTensor)


class ClassifierCalib:
    def __init__(self):
        self.temperature = 0
        self.model = MLP()
        self.epochs = 5
        self.batch_size = 32

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.model_path = './models/mnist_net.pth'

        self.trainloader = None
        self.testloader = None
        self.load_data()

    def load_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        train_set = MNIST(root='./data/mnist', train=True, download=True, transform=transform)
        test_set = MNIST(root='./data/mnist', train=False, download=True, transform=transform)

        self.trainloader = torch.utils.data.DataLoader(train_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=True, num_workers=0)

        self.testloader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=0)

    def train_model(self):

        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.cuda()
                inputs = inputs.squeeze().view(-1, 784)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 500 == 0:  # print every 500 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 500))
                    running_loss = 0.0

        print('Finished Training')

    def adjust_temperature(self):
        pass

    def train(self):
        self.train_model()
        self.save_model()
        self.test()

    def test(self):
        self.load_model()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data
                inputs = inputs.cuda()
                inputs = inputs.squeeze().view(-1, 784)

                outputs = self.model(inputs.cuda())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the %d test images: %d %%' % (
                total, 100 * correct / total))

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model = MLP()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    @staticmethod
    def plot_reliability_diagram(ground_truth, predicted):
        fop_uncalibrated, mpv_uncalibrated = calibration_curve(ground_truth, predicted, n_bins=10, normalize=True)
        # fop_calibrated, mpv_calibrated = calibration_curve(testy, yhat_calibrated, n_bins=10)

        # plot perfectly calibrated
        plt.plot([0, 1], [0, 1], linestyle='--', color='black')

        plt.plot(mpv_uncalibrated, fop_uncalibrated, marker='.')
        # pyplot.plot(mpv_calibrated, fop_calibrated, marker='.')
        plt.show()


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(784, 120)
        self.fc2 = torch.nn.Linear(120, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    fire.Fire(ClassifierCalib)
