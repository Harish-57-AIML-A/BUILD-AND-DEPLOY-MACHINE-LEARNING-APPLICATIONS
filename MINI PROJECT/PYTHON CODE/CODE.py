# dcgan_mnist_pytorch.py
# DCGAN for MNIST (28x28) with training tricks + hyperparameter hook
import os
import random
from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------
# Utilities & Hyperparameters
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
out_dir = './dcgan_runs'
os.makedirs(out_dir, exist_ok=True)

# Recommended starting hyperparameters (you can change)
default_config = {
    'z_dim': 100,
    'batch_size': 128,
    'lr': 0.0002,
    'beta1': 0.5,
    'epochs': 50,
    'img_size': 28,
    'ngf': 64,   # generator feature maps multiplier
    'ndf': 64,   # discriminator feature maps multiplier
    'save_every': 5,  # save images every N epochs
    'label_smooth': 0.9,  # real label = 0.9 (one-sided smoothing)
    'label_flip_prob': 0.03,  # small probability to flip labels
    'num_workers': 2
}

# ----------------------------
# Data
# ----------------------------
transform = transforms.Compose([
    transforms.Resize(default_config['img_size']),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # [-1,1]
])

dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# ----------------------------
# Model Definitions (DCGAN style)
# ----------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname or 'Linear' in classname:
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        except Exception:
            pass
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """
    Architecture:
    z -> linear -> (ngf*4 x 7 x 7) -> ConvTranspose(ngf*2) -> ConvTranspose(ngf) -> Conv2d(1)
    Produces 1 x 28 x 28
    """
    def __init__(self, z_dim=100, ngf=64):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Sequential(
            nn.Linear(z_dim, ngf*4*7*7),
            nn.BatchNorm1d(ngf*4*7*7),
            nn.ReLU(True)
        )
        self.net = nn.Sequential(
            # input is (ngf*4) x 7 x 7
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),  # -> 14x14
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=False),    # -> 28x28
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, 1, kernel_size=3, stride=1, padding=1),  # -> 1x28x28
            nn.Tanh()
        )

    def forward(self, z):
        # z: (B, z_dim)
        x = self.fc(z)
        x = x.view(x.size(0), -1, 7, 7)  # reshape to (B, ngf*4, 7, 7)
        # ensure the channel dim equals ngf*4
        # if reshape channel mismatch, debug: print(x.size())
        x = self.net(x)
        return x

class Discriminator(nn.Module):
    """
    Convolutional discriminator from 28x28 to a single logit.
    No sigmoid at end (we will use BCEWithLogitsLoss).
    """
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            # input 1 x 28 x 28
            nn.Conv2d(1, ndf, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(ndf*2*7*7, 1)  # output single logit
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# Training Routine
# ----------------------------
def train(config):
    # Set seed for reproducibility
    manual_seed = 999
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)

    # DataLoader
    pin_memory = True if device == 'cuda' else False
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['num_workers'], pin_memory=pin_memory)

    # Models
    G = Generator(z_dim=config['z_dim'], ngf=config['ngf']).to(device)
    D = Discriminator(ndf=config['ndf']).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    # Loss and optimizers
    criterion = nn.BCEWithLogitsLoss()
    opt_G = optim.Adam(G.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
    opt_D = optim.Adam(D.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))

    fixed_noise = torch.randn(64, config['z_dim'], device=device)

    step = 0
    for epoch in range(1, config['epochs']+1):
        loop = tqdm(loader, desc=f"Epoch [{epoch}/{config['epochs']}]")
        for real_imgs, _ in loop:
            real_imgs = real_imgs.to(device)
            bs = real_imgs.size(0)

            # Labels (with smoothing and occasional flips)
            real_label_val = config['label_smooth']
            fake_label_val = 0.0
            # Potential random flips
            if random.random() < config['label_flip_prob']:
                real_label_val, fake_label_val = 0.0, config['label_smooth']

            real_labels = torch.full((bs,1), real_label_val, device=device)
            fake_labels = torch.full((bs,1), fake_label_val, device=device)

            # ---- Train Discriminator ----
            D.zero_grad()
            # Real
            logits_real = D(real_imgs)
            loss_real = criterion(logits_real, real_labels)

            # Fake
            noise = torch.randn(bs, config['z_dim'], device=device)
            fake_imgs = G(noise)
            logits_fake = D(fake_imgs.detach())
            loss_fake = criterion(logits_fake, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()

            # ---- Train Generator ----
            G.zero_grad()
            # Recompute (or reuse) fake imgs; reuse here is fine
            logits_fake_for_G = D(fake_imgs)
            # Use real_labels as target so generator tries to make them appear real
            loss_G = criterion(logits_fake_for_G, real_labels)
            loss_G.backward()
            opt_G.step()

            step += 1

            loop.set_postfix(D_loss=loss_D.item(), G_loss=loss_G.item())

        # Save snapshots
        if epoch % config['save_every'] == 0 or epoch == config['epochs']:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise).cpu()
                samples = (samples * 0.5) + 0.5  # convert from [-1,1] to [0,1]
            grid = vutils.make_grid(samples, nrow=8, padding=2)
            vutils.save_image(grid, os.path.join(out_dir, f'epoch_{epoch:03d}.png'))
            torch.save({
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict()
            }, os.path.join(out_dir, f'checkpoint_epoch_{epoch:03d}.pth'))
            G.train()

    # Return final models/metrics if you like
    return G, D

# ----------------------------
# Example: single run with default config
# ----------------------------
if __name__ == '__main__':
    cfg = default_config.copy()
    # Option: adjust here before calling train
    G, D = train(cfg)

    # Display last saved image quickly (optional)
    import PIL.Image as Image
    path = os.path.join(out_dir, f'epoch_{cfg["epochs"]:03d}.png')
    if os.path.exists(path):
        img = Image.open(path)
        plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

