import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils
from torchvision.utils import save_image
import ConvAE
import matplotlib.pyplot as plt
import os

# Transforms
TRANSFORM_CIFAR10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

TRANSFORM_MNIST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Settings
settings = {
    'dataset': 'MNIST',
    'AE': 'ConvAE',
    'transform': TRANSFORM_MNIST,
    'train_batch_size': 128,
    'test_batch_size': 4,
    'lr': 1e-2,
    'epochs': 50,
    'weight_decay':1e-5,
}

def to_img(x):
    x = (x + 1.) * 0.5
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    model = ConvAE.ConvAutoEncoder()

    trainset = torchvision.datasets.MNIST(root='./datasets', download=True, transform=settings['transform'])

    train_data = utils.DataLoader(trainset, batch_size=settings['train_batch_size'], shuffle=True)

    distance = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])

    if torch.cuda.is_available():
        model.cuda()

    # Start training
    print('Start training!')
    for epoch in range(settings['epochs']):
        # Auto adjust the learning rate
        if epoch in [settings['epochs'] * 0.25, settings['epochs'] * 0.5]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        for img, _ in train_data:
            img = Variable(img.cuda())

            # forward
            _, output = model(img)
            loss = distance(output, img)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch[{}/{}], loss:{:.4f}'.format(epoch+1, settings['epochs'], loss.data.float()))

        if (epoch + 1) % 5 == 0:
            pic = to_img(output.cpu().data)
            if not os.path.exists('./encoder_{}'.format(settings['AE'])):
                os.mkdir('./encoder_{}'.format(settings['AE']))
            save_image(pic, './encoder_{}/image_{}.png'.format(settings['AE'], epoch + 1))

    torch.save(model, './autoencoder_{}'.format(settings['AE']))
