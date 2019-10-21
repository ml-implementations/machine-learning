"""
Implementation of BinaryConnect in PyTorch
"""

# TODO: Add hard-sigmoid (stochastic) binarization
# TODO: Add another network for CIFAR10 or other larger dataset
# TODO: New model architecture, maybe LeNet?
# TODO: Make Jupyter notebook

import fire
import torch
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.optim import Adam
from multiprocessing import set_start_method
from torch.nn import functional as F

torch.set_default_tensor_type(torch.cuda.FloatTensor)

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class BinaryConnect:
    def __init__(self):
        self.model = Model()
        self.batch_size = 32
        self.test_count = 9
        self.classes = 10

        self.train_mnist_dataloader = None
        self.test_mnist_dataloader = None
        self.mnist_epochs = 50
        self.model_opt = Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.loss = torch.nn.CrossEntropyLoss()
        self.prev_model = Model()

    def load_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor()]
        )

        train_set = MNIST(root='./data/mnist', train=True, download=True, transform=transform)
        self.train_mnist_dataloader = torch.utils.data.DataLoader(train_set,
                                                                  batch_size=self.batch_size,
                                                                  shuffle=True, num_workers=0)

        test_set = MNIST(root='./data/mnist', train=False, download=True, transform=transform)
        self.test_mnist_dataloader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size,
                                                                 shuffle=False, num_workers=0)

    def binarize(self, model):
        for param in model.parameters():
            ceiling = torch.ones(param.size())
            floor = torch.ones(param.size()) * -1
            param.data.copy_(torch.where(param.data > 0, ceiling, floor))

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
        for epoch in range(self.mnist_epochs):

            for i, data in enumerate(self.train_mnist_dataloader, 0):
                imgs, labels = data
                imgs = imgs.cuda().squeeze().view(-1, 784)

                # store previous params
                self.swap_params(self.model, self.prev_model)

                # binarize params
                self.binarize(self.model)

                # forward pass
                self.model_opt.zero_grad()
                preds = self.model(imgs)

                # calculate loss
                loss = self.loss(preds, labels)
                loss.backward()

                # swap-out binarized params
                self.swap_params(self.prev_model, self.model)

                # optimize
                self.model_opt.step()

                # clip params
                self.clip_params(self.model)

            # print results
            print("Epoch: ", epoch + 1, " Loss: ", loss.item())

        print('Finished Training')

    def test(self):
        self.load_model()
        sample_vector = torch.randn(self.batch_size, self.latent_vector_size)
        generated = self.model.decode(sample_vector)
        self.plot_results(generated)

    def load_model(self):
        self.model = Model()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(784, 120)
        self.fc2 = torch.nn.Linear(120, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# wrapping to avoid Windows 10 error
def main():
    fire.Fire(BinaryConnect)


if __name__ == "__main__":
    main()
