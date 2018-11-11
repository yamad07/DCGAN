import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.utils as vutils
import src.utils.dcgan as utils

class Trainer(object):

    def __init__(self, discriminator, generator, batch_size, img_size, dim_z, beta1):
        self.discriminator = discriminator
        self.generator = generator
        self.batch_size = batch_size
        self.img_size = img_size
        self.dim_z = dim_z
        self.beta1 = beta1

    def train(self, lr, epoch, dataloader):
        self.generator.apply(utils.weights_init)
        self.discriminator.apply(utils.weights_init)

        criterion = nn.BCELoss()
        input = torch.FloatTensor(self.batch_size, 3, self.img_size, self.img_size)
        noise = torch.FloatTensor(self.batch_size, self.dim_z, 1, 1)
        fixed_noise = torch.FloatTensor(self.batch_size, self.dim_z, 1, 1).normal_(0, 1)
        label = torch.FloatTensor(self.batch_size)
        real_label = 1
        fake_label = 0

        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(self.beta1, 0.999))
        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(self.beta1, 0.999))

        for e in range(epoch):
            for i, data in enumerate(dataloader, 0):
                self.discriminator.zero_grad()
                real, _ = data
                batch_size = real.size(0)

                input.resize_as_(real).copy_(real)
                label.resize_(batch_size).fill_(real_label)

                output = self.discriminator(input)
                real_loss = criterion(output, label)
                real_loss.backward()
                d_x = output.data.mean()

                # train with fake
                noise.resize_(batch_size, self.dim_z, 1, 1).normal_(0, 1)
                fake = self.generator(noise)
                label = label.fill_(fake_label)
                output = self.discriminator(fake.detach())
                fake_loss = criterion(output, label)
                fake_loss.backward()
                d_g_z1 = output.data.mean()
                loss = real_loss + fake_loss
                optimizer_d.step()


                optimizer_g.zero_grad()
                label = label.fill_(real_label)  # fake labels are real for generator cost
                output = self.discriminator(fake)
                g_loss = criterion(output, label)
                g_loss.backward()
                d_g_z2 = output.data.mean()
                optimizer_g.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (e, epoch, i, len(dataloader),
                         loss.item(), g_loss.item(), d_x, d_g_z1, d_g_z2))
                if i % 100 == 0:
                    vutils.save_image(real,
                            'results/real_samples.png',
                            normalize=True)
                    fake = self.generator(fixed_noise)
                    vutils.save_image(fake.data,
                            'results/fake_samples_epoch_%03d.png' % epoch,
                            normalize=True)
