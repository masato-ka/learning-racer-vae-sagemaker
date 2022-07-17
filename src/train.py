import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms
import numpy as np
import argparse
import os
import json


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 3, 8)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=6144, z_dim=32):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(),
        )

        self.out1 = nn.Sequential(nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2),
                                  nn.Sigmoid(),
                                  )
        self.out2 = nn.Sequential(nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2),
                                  nn.Sigmoid(),
                                  )

    def reparameterize(self, mu, logvar):

        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())#.to(device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)  # F.softplus(self.fc2(h))
        if self.training:
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        z = mu
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        x = self.decoder(z)
        mu_y = self.out1(x)
        sigma_y = self.out2(x)
        return mu_y, sigma_y

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        mu_y, sigma_y = self.decode(z)
        return mu_y, sigma_y, mu, logvar

    def loss_fn(self, image, mu_y, sigma_y, mean, logvar):
        m_vae_loss = (image - mu_y) ** 2 / sigma_y
        m_vae_loss = 0.5 * torch.sum(m_vae_loss)
        a_vae_loss = torch.log(2 * 3.14 * sigma_y)
        a_vae_loss = 0.5 * torch.sum(a_vae_loss)
        KL = -0.5 * torch.sum((1 + logvar - mean.pow(2) - logvar.exp()), dim=0)
        KL = torch.mean(KL)
        return KL + m_vae_loss + a_vae_loss

def create_model(device, z_dim=32):
    model = VAE(image_channels=3, z_dim=z_dim).to(device)
    return model

def load_dataset(train_data_path, test_data_path, bs=64):
    """
    train_data_path is prepare to dataset_root/dataset/*.jpg
    and train_data_path seted to absoluto or relevant path to dataset_root.
    :param train_data_path:
    :param test_data_path:
    :param bs:
    :return:
    """
    bs = 64
    dataset = datasets.ImageFolder(root=train_data_path, transform=transforms.Compose([
        torchvision.transforms.Resize((120, 160)),
        torchvision.transforms.Lambda(lambda x: x.crop((0, 40, 160, 120))),
        transforms.ToTensor(),
    ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True,  num_workers=2, pin_memory=True)
    len(dataset.imgs), len(dataloader)
    return dataloader


def _parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--use-cuda', type=bool, default=False)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    return parser.parse_known_args()

def train(device, model, dataloader, logdir, epochs=100, lr=1e-3):
    writer = SummaryWriter('/opt/ml/output/tensorboard')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        losses = []
        grid = None
        for idx, (images, _) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad()
            mu_y, sigma_y, mu, logvar = model(images)
            loss = model.loss_fn(images, mu_y, sigma_y, mu, logvar)
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
            grid = torchvision.utils.make_grid(mu_y)
            grid_sigma = torchvision.utils.make_grid(sigma_y)
            writer.add_image('Image/reconst', grid, epoch)
            writer.add_image('Image/sigma', grid_sigma, epoch)
            writer.add_scalar('Loss/train',np.average(losses), epoch)
        print("EPOCH: {} loss: {}".format(epoch+1, np.average(losses)))
    return model

def save_model_artifact(save_path):
    pass


def main():
    args, _ = _parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    model = create_model(device)
    dataset = load_dataset(args.train, args.train)
    model = train(device, model, dataset, logdir=args.model_dir, epochs=args.epochs)

    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f, _use_new_zipfile_serialization=False)

if __name__ == '__main__':
    main()