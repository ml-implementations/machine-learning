from PIL import Image
from torch import optim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import fire
import numpy as np

'''
A Neural Algorithm of Artistic Style implementation

https://arxiv.org/pdf/1508.06576.pdf
'''


class NAAS:
    def __init__(self):
        # model
        self.vgg16 = None
        self.vgg16_pretrained_weights = 'models/vgg_conv_weights.pth'
        self.vgg16_mean = np.array([0.40760392, 0.45795686, 0.48501961])

        # transformations
        self.transforms_pre = None
        self.transforms_post = None

        # device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # images
        self.content = None
        self.style = None
        self.opt = None

        # image characteristics
        self.images_dir = 'data/images/'
        self.images_size = (256, 256)
        self.images_shape = (3, *self.images_size)

        # training
        self.optimizer = None
        self.iterations = 100
        self.MSE = nn.MSELoss()
        self.content_loss_layers = ['r42']
        self.style_loss_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
        self.style_loss_layers_weights = [1e3/n**2 for n in [64, 128, 256, 512, 512]]
        self.total_loss_layers = self.content_loss_layers + self.style_loss_layers

    def load_vgg_16(self):
        self.vgg16 = VGG()
        self.vgg16.load_state_dict(torch.load(self.vgg16_pretrained_weights))
        self.vgg16 = self.vgg16.to(self.device)

    def define_transformations_preprocessing(self):
        torch_transforms = list()
        # the pre-trained models require the images to be normalized in a certain way
        torch_transforms.append(transforms.Resize(self.images_size))
        torch_transforms.append(transforms.ToTensor())
        torch_transforms.append(transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]))
        torch_transforms.append(transforms.Normalize(mean=self.vgg16_mean, std=[1, 1, 1]))
        torch_transforms.append(transforms.Lambda(lambda x: x.mul_(255)))
        self.transforms_pre = transforms.Compose(torch_transforms)

    def define_transformations_postprocessing(self):
        torch_transforms = list()
        # the postprocessing transforms to bring images to back to their original form
        torch_transforms.append(transforms.Lambda(lambda x: x.mul_(1./255)))
        torch_transforms.append(transforms.Normalize(mean=-self.vgg16_mean, std=[1, 1, 1]))
        torch_transforms.append(transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]))
        torch_transforms.append(transforms.ToPILImage())
        self.transforms_post = transforms.Compose(torch_transforms)

    def load_images_and_transform(self):
        self.content = self.transforms_pre(Image.open(self.images_dir + 'content.jpg')).to(self.device)
        self.style = self.transforms_pre(Image.open(self.images_dir + 'style.jpg')).to(self.device)
        self.opt = torch.tensor(np.random.normal(size=self.images_shape), requires_grad=True,
                                dtype=torch.float, device=self.device)

    @staticmethod
    def calculate_gram_matrix(mat):
        b, c, w, h = mat.size()
        mat = mat.view(b, c, h * w)
        mat = torch.bmm(mat, mat.transpose(1, 2))
        mat = mat / (h * w)
        return mat

    def train(self):
        self.load_vgg_16()
        self.define_transformations_preprocessing()
        self.define_transformations_postprocessing()
        self.load_images_and_transform()
        self.optimizer = optim.LBFGS([self.opt])

        for i in range(self.iterations):

            def closure():
                total_loss = list()

                content_layers = self.vgg16.forward(torch.unsqueeze(self.content, dim=0), self.content_loss_layers)
                opt_layers = self.vgg16.forward(torch.unsqueeze(self.opt, dim=0), self.content_loss_layers)

                for j in range(len(self.content_loss_layers)):
                    mse_loss = self.MSE(opt_layers[j], content_layers[j])
                    total_loss.append(mse_loss)

                style_layers = self.vgg16.forward(torch.unsqueeze(self.style, dim=0), self.style_loss_layers)
                opt_layers = self.vgg16.forward(torch.unsqueeze(self.opt, dim=0), self.style_loss_layers)

                for j in range(len(self.style_loss_layers)):
                    gram_matrix_style = self.calculate_gram_matrix(style_layers[j])
                    gram_matrix_opt = self.calculate_gram_matrix(opt_layers[j])
                    mse_loss = self.MSE(gram_matrix_opt, gram_matrix_style)
                    total_loss.append(self.style_loss_layers_weights[j] * mse_loss)

                self.optimizer.zero_grad()
                total_loss = sum(total_loss)
                total_loss.backward()

                print("Iteration: ", i + 1, "\tLoss: ", total_loss)
                img = self.transforms_post(self.opt.clone().cpu())
                img.save(self.images_dir + 'opt_{}.jpg'.format(i))

                return total_loss

            self.optimizer.step(closure)



'''
VGG Model: Very Deep Convolutional Networks for Large-Scale Image Recognition

https://arxiv.org/pdf/1409.1556.pdf
implementation from: https://github.com/FarisNolan/Neural_Algorithm_Artistic_Style/blob/master/N_A_A_S.py
'''


class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # CONV LAYERS
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # HANDLE POOLING OPTIONS
        # MAX POOLING
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # AVERAGE POOLING
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        # FORWARD PROP

    def forward(self, x, out_keys):
        out = dict()

        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])

        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])

        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])

        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])

        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])

        return [out[key] for key in out_keys]


if __name__ == '__main__':
    fire.Fire(NAAS)
