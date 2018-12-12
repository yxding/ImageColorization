
# coding: utf-8

# In[ ]:


import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from models import *
from dataloader import *

import warnings
warnings.simplefilter("ignore")

os.makedirs('sample', exist_ok=True)


# In[ ]:


cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)


# In[ ]:


# Configure data loader
dataset = Dataset(root_dir= './data')

dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# In[ ]:


for epoch in range(20):
    for i, data in enumerate(dataloader):

        gray_imgs = data[0].type(Tensor)
        color_imgs = data[1].type(Tensor)

        # Adversarial ground truths
        valid = Variable(Tensor(color_imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(color_imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(color_imgs.type(Tensor), requires_grad=False)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(gray_imgs)
        
        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gray_imgs)
        g_loss = adversarial_loss(validity, fake)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs, gray_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gray_imgs), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, 20, i, len(dataloader),
                                                            d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % 1000 == 0:
            save_image(gen_imgs.data[:25], 'sample/%d.png' % batches_done, nrow=5, normalize=True)

