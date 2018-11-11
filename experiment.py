from src.models.dcgan import Generator, Discriminator
from src.trainer.dcgan import Trainer

import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

dataroot = './data/'
img_size = 64
dim_z = 100
beta1 = 0.5
batch_size = 10

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Scale(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)

trainer = Trainer(
        discriminator=Discriminator(),
        generator=Generator(),
        batch_size=batch_size,
        img_size=img_size,
        dim_z=dim_z,
        beta1=beta1
        )
trainer.train(
        lr=0.0001,
        epoch=100,
        dataloader=dataloader
        )
