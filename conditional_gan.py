import numpy as np
import fire
import cv2
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.optim import Adam
from multiprocessing import set_start_method

torch.set_default_tensor_type(torch.cuda.FloatTensor)

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class CGAN:
    def __init__(self):
        self.discriminator = Discriminator()
        self.generator = Generator()
        self.gan = None
        self.gan_input = 100
        self.batch_size = 32
        self.test_count = 9
        self.classes = 10

        self.train_mnist_dataloader = None
        self.test_mnist_dataloader = None
        self.mnist_epochs = 50
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.generator_opt = Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.loss = nn.BCELoss()

        self.generator_model_path = 'models/cgan.hdf5'

    def load_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor()]
        )

        train_set = MNIST(root='./data/mnist', train=True, download=True, transform=transform)
        self.train_mnist_dataloader = torch.utils.data.DataLoader(train_set,
                                                                  batch_size=self.batch_size,
                                                                  shuffle=True, num_workers=1)

        test_set = MNIST(root='./data/mnist', train=False, download=True, transform=transform)
        self.test_mnist_dataloader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size,
                                                                 shuffle=False, num_workers=1)

    def train(self):
        self.load_data()
        for epoch in range(self.mnist_epochs):

            for i, data in enumerate(self.train_mnist_dataloader, 0):
                real, real_labels = data
                real_target = torch.ones(size=(self.batch_size, 1), requires_grad=False) - 0.1
                fake_target = torch.zeros(size=(self.batch_size, 1), requires_grad=False)

                # generator update
                self.generator_opt.zero_grad()
                noise = torch.tensor(np.random.normal(0, 1, (self.batch_size, 100)), dtype=torch.float)
                noise_labels = torch.tensor(np.random.randint(0, self.classes, size=(self.batch_size, 1)),
                                            dtype=torch.long)
                fake = self.generator(noise, noise_labels)
                g_loss = self.loss(self.discriminator(fake, noise_labels), real_target)
                g_loss.backward()
                self.generator_opt.step()

                # discriminator update
                self.discriminator_opt.zero_grad()
                real_loss = self.loss(self.discriminator(real.cuda().detach(), real_labels), real_target)
                fake_loss = self.loss(self.discriminator(fake.detach(), noise_labels), fake_target)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.discriminator_opt.step()

                print("Generator Loss: ", g_loss)
                print("Discriminator Loss: ", d_loss)

            # print results
            self.sample_and_save_gan(epoch)
            print("Epoch: ", epoch + 1)

        print('Finished Training')

    def sample_and_save_gan(self, epoch):
        noise = torch.randn(size=(1, self.gan_input))
        labels = torch.randint(self.classes - 1, size=(1, 1))
        img = self.generator(noise, labels)
        img = img.cpu().detach().numpy()
        img = np.squeeze(img, axis=0)
        img = np.squeeze(img, axis=0)
        img = img * 255.0
        print(img)
        cv2.imwrite('gan_generated/img_{}.png'.format(epoch), img)
        torch.save(self.generator.state_dict(), self.generator_model_path)

    def plot_results(self, generated):
        fig = plt.figure(figsize=(28, 28))
        columns = np.sqrt(self.test_count)
        rows = np.sqrt(self.test_count)
        generated = generated.cpu().detach().numpy()
        generated = np.squeeze(generated, axis=1)
        generated = generated * 255.0
        for i in range(1, int(columns) * int(rows) + 1):
            fig.add_subplot(rows, columns, i)
            plt.imshow(generated[i - 1], cmap='gray_r')
        plt.show()

    def test(self):
        self.load_generator()
        noise = torch.randn(size=(self.test_count, self.gan_input))
        labels = torch.randint(self.classes - 1, size=(self.test_count, 1))
        print(labels)
        generated = self.generator(noise, labels)
        self.plot_results(generated)

    def load_generator(self):
        self.generator = Generator()
        self.generator.load_state_dict(torch.load(self.generator_model_path))
        self.generator.eval()


class Discriminator(nn.Module):
    def __init__(self, n_classes=10):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.drop_out1 = nn.Dropout(0.4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.drop_out2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(7 * 7 * 64, 1)
        self.sigmoid = nn.Sigmoid()

        self.embedding1 = nn.Embedding(n_classes, 50)
        self.fc1_label = nn.Linear(50, 784)

    def forward(self, x, y):
        y = self.embedding1(y)
        y = self.fc1_label(y)
        y = y.view(-1, 1, 28, 28)

        x = torch.cat([x, y], dim=1)

        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.drop_out1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.drop_out2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.sigmoid(self.fc1(x))

        return x


class Generator(nn.Module):
    def __init__(self, n_classes=10):
        super(Generator, self).__init__()
        self.fc1_out = 128 * 7 * 7
        self.fc1 = nn.Linear(100, self.fc1_out)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv1 = nn.ConvTranspose2d(129, 128, kernel_size=4, stride=2, padding=1)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

        self.embedding1 = nn.Embedding(n_classes, 50)
        self.fc1_label = nn.Linear(50, 49)

    def forward(self, x, y):
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = x.view(-1, 128, 7, 7)

        y = self.embedding1(y)
        y = self.fc1_label(y)
        y = y.view(-1, 1, 7, 7)

        x = torch.cat([x, y], dim=1)

        x = self.conv1(x)
        x = self.leaky_relu2(x)
        x = self.conv2(x)
        x = self.leaky_relu3(x)
        x = self.sigmoid(self.conv3(x))

        return x


if __name__ == "__main__":
    fire.Fire(CGAN)
