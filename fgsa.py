# Fast Gradient Sign Attack in PyTorch
# Reference: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

epsilons = [0, .05, .1, .15, .2, .25, .3]
use_cuda = True
pretrained_model = "data/lenet_mnist_model.pth"


# model under attack
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=1, shuffle= True
)

# run on GPU if available
print('CUDA available: ', torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# init model
model = Net().to(device)

# load pre-trained model
model.load_state_dict(torch.load(pretrained_model, map_location="cpu"))

# set model to eval mode
model.eval()


# FGSM attacker
def fgsm_attack(image, epsilon, data_grad):
    # element-wise sign of the data gradients
    sign_data_grad = data_grad.sign()
    # perturbe image by adding noise to each pixel
    perturbed_image = image + epsilon * sign_data_grad
    # clip to maintain [0, 1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# test adversarial attack
def test(model, device, test_loader, epsilon):
    # accuracy count
    correct = 0
    adv_examples = []

    # loop over all examples in test set
    for data, target in test_loader:

        # send data to device (CPU or GPU)
        data, target = data.to(device), target.to(device)

        # set input data require grad to true
        data.requires_grad = True

        # run model and get pre-attack prediction (index of max log-probability)
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        # if init_pred != target, no need for attack and continue
        if init_pred.item() != target.item():
            continue

        # calculate loss
        loss = F.nll_loss(output, target)

        # zero all existing gradients
        model.zero_grad()

        # calculate grads in backward pass
        loss.backward()

        # collect grads
        data_grad = data.grad.data

        # make attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # predict sample again, this time after attack
        output = model.forward(perturbed_data)

        # check if attack successful
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            # case if epsilon is 0 (no noise added), save examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

        else:
            # save adversarial examples for vis
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\t Test Accuracy: {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    return final_acc, adv_examples


# run adversarial attack
accuracies = []
examples = []

for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)


# plot results
plt.figure(figsize=(5, 5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs. Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

# plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()




