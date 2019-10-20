"""
Implementation of a Variational Auto-encoder in PyTorch
"""

import numpy as np
import fire
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.optim import Adam
from torch.distributions.normal import Normal
from multiprocessing import set_start_method
from torch.nn import functional as F

torch.set_default_tensor_type(torch.cuda.FloatTensor)

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class VAE:
    def __init__(self):
        self.model = Model()
        self.latent_vector_size = 20
        self.batch_size = 32
        self.test_count = 9
        self.classes = 10

        self.train_mnist_dataloader = None
        self.test_mnist_dataloader = None
        self.mnist_epochs = 50
        self.model_opt = Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.generated_loss = torch.nn.BCELoss()
        self.latent_loss = torch.nn.KLDivLoss()
        self.dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        self.model_path = 'models/vae.hdf5'

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

    def sample_and_save_image(self, epoch):
        sample_vector = torch.randn(1, self.latent_vector_size)
        img = self.model.decode(sample_vector)
        img = img.view(-1, 28, 28)
        img = img.cpu().detach().numpy()
        img = np.squeeze(img, axis=0)
        img = img * 255.0
        cv2.imwrite('vae_generated/img_{}.png'.format(epoch), img)
        torch.save(self.model.state_dict(), self.model_path)

    def plot_results(self, generated):
        fig = plt.figure(figsize=(28, 28))
        columns = np.sqrt(self.test_count)
        rows = np.sqrt(self.test_count)
        generated = generated.view(-1, 28, 28)
        generated = generated.cpu().detach().numpy()
        generated = generated * 255.0
        for i in range(1, int(columns) * int(rows) + 1):
            fig.add_subplot(rows, columns, i)
            plt.imshow(generated[i - 1], cmap='gray_r')
        plt.show()

    def train(self):
        self.load_data()
        self.model.train()
        for epoch in range(self.mnist_epochs):

            for i, data in enumerate(self.train_mnist_dataloader, 0):
                real, real_labels = data
                real = real.cuda().squeeze().view(-1, 784)

                # run encoder
                self.model_opt.zero_grad()
                mean, logvar = self.model.encode(real)
                std = torch.exp(0.5 * logvar)

                # sample unit gaussian vector
                sample_vector = torch.randn_like(std)
                latent_vector = (sample_vector * std) + mean

                # calculate loss
                generated_loss = self.generated_loss(self.model.decode(latent_vector), real)
                latent_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                total_loss = generated_loss + latent_loss
                total_loss.backward()

                # optimize
                self.model_opt.step()

            # print results
            self.sample_and_save_image(epoch)
            print("Epoch: ", epoch + 1, " Loss: ", total_loss)

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
    def __init__(self, n_classes=10):
        super(Model, self).__init__()

        self.fc1 = torch.nn.Linear(784, 400)
        self.fc21 = torch.nn.Linear(400, 20)
        self.fc22 = torch.nn.Linear(400, 20)
        self.fc3 = torch.nn.Linear(20, 400)
        self.fc4 = torch.nn.Linear(400, 784)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        mean = self.fc21(x)
        logvar = self.fc22(x)

        return mean, logvar

    def decode(self, x):
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))

        return x


# wrapping to avoid Windows 10 error
def main():
    fire.Fire(VAE)


if __name__ == "__main__":
    main()
