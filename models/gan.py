import time
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
from .utils import to_timedict


class Gan:
    def __init__(self, config, dataloader, generator, discriminator, device=None):
        self.config = config
        self.dataloader = dataloader
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.g_net = generator.to(self.device)
        self.d_net = discriminator.to(self.device)
        # optimizers
        self.g_optimizer = optim.Adam(self.g_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.d_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
        # loss criterion
        self.criterion = nn.BCELoss()
        # dummies
        self.ones = torch.ones(config.batch_size).to(self.device)
        self.zeros = torch.zeros(config.batch_size).to(self.device)
        self.fixed_z = (
            torch.rand(config.n_show * config.n_show, config.hidden_size, 1, 1)
            .uniform_(-1, 1)
            .to(self.device)
        )

    def train_g(self, x):
        self.g_net.zero_grad()
        z = (
            torch.rand(self.config.batch_size, self.config.hidden_size, 1, 1)
            .uniform_(-1, 1)
            .to(self.device)
        )
        x_fake = self.g_net(z)
        pred_fake = self.d_net(x_fake)
        # generated images with labels=1, as if they're real images
        g_loss = self.criterion(pred_fake.squeeze(), self.ones)
        g_loss.backward()
        self.g_optimizer.step()
        return g_loss.item()

    def train_d(self, x):
        self.d_net.zero_grad()
        z = (
            torch.rand(self.config.batch_size, self.config.hidden_size, 1, 1)
            .uniform_(-1, 1)
            .to(self.device)
        )
        with torch.no_grad():
            # avoid building computation graph for G, i.e. x_fake has no grad
            x_fake = self.g_net(z)
        pred_fake = self.d_net(x_fake)
        # generated images with labels=0
        d_loss_fake = self.criterion(pred_fake.squeeze(), self.zeros)
        pred_real = self.d_net(x)
        # real images with labels=1
        d_loss_real = self.criterion(pred_real.squeeze(), self.ones)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        return d_loss.item()

    def train_one_epoch(self):
        d_losses = []
        g_losses = []
        weights = []
        for (x, _) in tqdm(self.dataloader):
            if x.size(0) != self.config.batch_size:
                continue
            x = x.to(self.device)
            d_losses.append(self.train_d(x))
            g_losses.append(self.train_g(x))
            weights.append(len(x))
        d_loss = np.average(d_losses, weights=weights)
        g_loss = np.average(g_losses, weights=weights)
        return d_loss, g_loss

    def train(self):
        for epoch in tqdm(range(1, self.config.num_epochs + 1)):
            start_time = time.time()
            d_loss, g_loss = self.train_one_epoch()
            end_time = time.time()
            if epoch % self.config.report_freq == 0:
                self.report(epoch, start_time, end_time, d_loss, g_loss)

    def report(self, epoch, start_time, end_time, d_loss, g_loss):
        t = to_timedict(end_time - start_time)
        msg = "| epoch: {:03} | d_loss: {:.03f} | g_loss: {:.03f} | elapsed: {}m {}s".format(
            epoch, d_loss, g_loss, t["mins"], t["secs"]
        )
        print(msg)

        with torch.no_grad():
            generated_images = self.g_net(self.fixed_z)
        generated_images = generated_images.reshape(
            -1, 3, self.config.image_size, self.config.image_size
        ).cpu()
        grid = torchvision.utils.make_grid(generated_images, nrow=self.config.n_show)
        grid = grid.permute(1, 2, 0)
        plt.imshow(grid)
        plt.show()

    def checkpoint(self):
        # TODO
        pass

    def vis(self):
        # TODO
        pass
