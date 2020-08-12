from fire import Fire
import matplotlib.pyplot as plt
import torch
from torch.nn import Module, Linear, MSELoss
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor


class AEModel(Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.enc_1 = Linear(in_features=784, out_features=128)
        self.enc_2 = Linear(in_features=128, out_features=32)
        self.dec_1 = Linear(in_features=32, out_features=128)
        self.dec_2 = Linear(in_features=128, out_features=784)

    def forward(self, x):
        # encode
        x = torch.relu(self.enc_1(x))
        latent_embedding = torch.relu(self.enc_2(x))

        # decode
        y = torch.relu(self.dec_1(latent_embedding))
        reconstructed = torch.relu(self.dec_2(y))

        return reconstructed


class Autoencoder():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AEModel().to(self.device)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss = MSELoss()
        self.epochs = 20

        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        transform = Compose([ToTensor()])

        train_data = MNIST(root="~/torch_datasets",
                           train=True,
                           transform=transform,
                           download=True)
        test_data = MNIST(root="~/torch_datasets",
                          train=False,
                          transform=transform,
                          download=True)

        self.train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    @staticmethod
    def plot_results(X, y):
        plt.close('all')
        plt.imshow(y.cpu().detach().numpy(), cmap='gray')
        plt.show()

    def train(self):
        self.load_data()
        sample_X, sample_y = None, None
        for i in range(self.epochs):
            loss = 0
            for X, _ in self.train_loader:
                X = X.view(-1, 784).to(self.device)

                self.opt.zero_grad()

                outputs = self.model(X)

                train_loss = self.loss(outputs, X)

                train_loss.backward()

                self.opt.step()

                loss += train_loss.item()

                sample_X = X[0, :].view(28, 28)
                sample_y = outputs[0, :].view(28, 28)

            self.plot_results(sample_X, sample_y)
            loss = loss / len(self.train_loader)
            print("Epoch: {}/{}, Loss: {}".format(i + 1, self.epochs, loss))


if __name__ == "__main__":
    Fire(Autoencoder)
