"""
Implementation of BinaryConnect in PyTorch
"""
import fire
import torch
import time
from torchvision.transforms import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.optim import Adam
from multiprocessing import set_start_method
from torch.nn import functional as F

torch.set_default_tensor_type(torch.cuda.FloatTensor)

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class BinaryConnect:
    def __init__(self, dataset='mnist', compress=True, stochastic_bin=False):
        self.compress = compress
        self.dataset = dataset
        self.stochastic_bin = stochastic_bin
        self.batch_size = 32
        self.test_count = 9
        self.train_dataloader = None
        self.test_dataloader = None
        self.mnist_epochs = 50
        self.loss = torch.nn.CrossEntropyLoss()

        if self.dataset == 'mnist':
            self.model = MNIST_Model()
            self.prev_model = MNIST_Model()
        else:
            self.model = CIFAR_Model()
            self.prev_model = CIFAR_Model()
        self.model_opt = Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def load_data(self):
        if self.dataset == 'mnist':
            transform = transforms.Compose(
                [transforms.ToTensor()]
            )
            train_set = MNIST(root='./data/' + self.dataset, train=True, download=True, transform=transform)
            test_set = MNIST(root='./data/' + self.dataset, train=False, download=True, transform=transform)
        else:
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_set = CIFAR10(root='./data/' + self.dataset, train=True, download=True, transform=transform)
            test_set = CIFAR10(root='./data/' + self.dataset, train=False, download=True, transform=transform)

        self.train_dataloader = torch.utils.data.DataLoader(train_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=True, num_workers=0)

        self.test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=0)

    def hard_sigmoid(self):
        pass

    def binarize(self, model):
        for param in model.parameters():
            ceiling = torch.ones(param.size())
            floor = torch.ones(param.size()) * -1
            thresh = 0
            # TODO: Add hard-sigmoid (stochastic) binarization
            if self.stochastic_bin:
                pass
            else:
                pass
            param.data.copy_(torch.where(param.data > thresh, ceiling, floor))

    def swap_params(self, source, target):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def clip_params(self, model):
        for param in model.parameters():
            ceiling = torch.ones(param.size())
            floor = torch.ones(param.size()) * -1
            param.data.copy_(torch.where(param.data > 1, ceiling, param.data))
            param.data.copy_(torch.where(param.data < -1, floor, param.data))

    def train(self):
        self.load_data()
        self.model.train()
        loss_time = []
        optim_time = []
        for epoch in range(self.mnist_epochs):

            for i, data in enumerate(self.train_dataloader, 0):
                imgs, labels = data
                imgs = imgs.cuda()
                if self.dataset == 'mnist':
                    imgs = imgs.squeeze().view(-1, 784)

                if self.compress:
                    # store previous params
                    self.swap_params(self.model, self.prev_model)

                    # binarize params
                    self.binarize(self.model)

                # forward pass
                self.model_opt.zero_grad()
                preds = self.model(imgs)

                # calculate loss
                start = time.time()
                loss = self.loss(preds, labels)
                loss.backward()
                loss_time.append(time.time() - start)

                if self.compress:
                    # swap-out binarized params
                    self.swap_params(self.prev_model, self.model)

                # optimize
                start = time.time()
                self.model_opt.step()
                optim_time.append(time.time() - start)

                if self.compress:
                    # clip params
                    self.clip_params(self.model)

            # print results
            avg_loss_time = round(sum(loss_time)/len(loss_time), 7)
            avg_optim_time = round(sum(optim_time)/len(optim_time), 7)

            print("Epoch: ", epoch + 1, " Loss: ", loss.item())
            print("Loss and Optim took (average) {}, {} seconds to calculate respectively."
                  .format(avg_loss_time, avg_optim_time))

        print('Finished Training')

    def test(self):
        self.load_model()
        sample_vector = torch.randn(self.batch_size, self.latent_vector_size)
        generated = self.model.decode(sample_vector)
        self.plot_results(generated)

    def load_model(self):
        self.model = MNIST_Model()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()


class MNIST_Model(torch.nn.Module):
    def __init__(self):
        super(MNIST_Model, self).__init__()
        self.fc1 = torch.nn.Linear(784, 120)
        self.fc2 = torch.nn.Linear(120, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CIFAR_Model(torch.nn.Module):
    def __init__(self):
        super(CIFAR_Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# wrapping to avoid Windows 10 error
def main():
    fire.Fire(BinaryConnect)


if __name__ == "__main__":
    main()
