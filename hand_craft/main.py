import torch
from torch import nn
from util import ResidualConvBlock, UnetDown, UnetUp, EmbedFC, plot_sample, CustomDataset
from dataset import SpriteDataset
import numpy as np
import torch.optim as optim
from pathlib import Path


TRAIN = False


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, pixel_height=28):  # cfeat - context features
        super(ContextUnet, self).__init__()

        # number of input channels, number of intermediate feature maps and number of classes
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = pixel_height  #assume h == w. must be divisible by 4, so 28,24,20,16...

        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Initialize the down-sampling path of the U-Net with two levels
        self.down1 = UnetDown(n_feat, n_feat)        # down1 #[10, 256, 32, 32]
        self.down2 = UnetDown(n_feat, 2 * n_feat)    # down2 #[10, 256, 4,  4]

        # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.to_vec = nn.Sequential(nn.AvgPool2d(int(pixel_height / 2**2)), nn.GELU())

        # Embed the timestep and context labels with a one-layer fully connected neural network
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2*n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1*n_feat)

        # Initialize the up-sampling path of the U-Net with three levels
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h//4, self.h//4), # up-sample 
            nn.GroupNorm(8, 2 * n_feat), # normalize                        
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1), # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.GroupNorm(8, n_feat), # normalize
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1), # map to same number of channels as input
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat)      : time step
        c : (batch, n_classes)    : context label
        """
        # x is the input image, c is the context label, t is the timestep, context_mask says which samples to block the context on

        # pass the input image through the initial convolutional layer
        x = self.init_conv(x)
        # pass the result through the down-sampling path
        down1 = self.down1(x)       #[10, 256, 8, 8]
        down2 = self.down2(down1)   #[10, 256, 4, 4]
        
        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down2)
        
        # mask out context if context_mask == 1
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)
            
        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)     # (batch, 2*n_feat, 1,1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        #print(f"uunet forward: cemb1 {cemb1.shape}. temb1 {temb1.shape}, cemb2 {cemb2.shape}. temb2 {temb2.shape}")


        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out



# ============= Parameters ==============
time_steps = 500
batch_size = 100
# n_feat = 64
n_feat = 512
n_cfeat = 5
pixel_height = 32
# pixel_height = 16
n_epoch = 50
lr = 1e-3
# ========================================

# ============= Dataset ==================
dataset = SpriteDataset()
# dataset = CustomDataset(
#     data_dir=Path("dataset/sprites/"),
# )
# =========================================

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ContextUnet(
    in_channels=3,
    n_feat=n_feat,
    n_cfeat=n_cfeat,
    pixel_height=pixel_height).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

model.train()

beta = (0.02 - 0.0001) * torch.linspace(0, 1, time_steps + 1, device=device)
alpha = 1 - beta
alpha_bar = torch.cumsum(alpha.log(), dim=0).exp()
alpha_bar[0] = 1.0


def add_noise(samples, t, noise):
    return alpha_bar.sqrt()[t, None, None, None] * samples + (1-alpha_bar[t, None, None, None]).sqrt() * noise


if TRAIN:
    # Forward pass example
    for epoch in range(n_epoch):
        optimizer.param_groups[0]['lr'] = lr*(1-epoch/n_epoch)
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            t = torch.randint(1, time_steps + 1, (images.shape[0],)).to(device)
            noise = torch.randn_like(images)
            noised_images = add_noise(images, t, noise)

            optimizer.zero_grad()
            outputs = model(noised_images, t / time_steps, c=None)
            loss = criterion(noise, outputs)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{n_epoch}], Step [{i}/{len(data_loader)}], Loss: {loss.item():.4f}")

    # save checkpoint
    torch.save(model.state_dict(), "context_unet.pth")
else:
    # Load checkpoint
    model.load_state_dict(torch.load("context_unet.pth"))
    print("Model loaded from checkpoint.")
    # Set the model to evaluation mode
    model.eval()



# Sampling
n_sample = 8
save_rate = 20
save_dir = "generated_images/"
samples_xt = torch.randn(n_sample, 3, pixel_height, pixel_height).to(device)
model = model.eval()


def ddpm_sample_step(x_t, t, model, T):
    # t: scalar timestep (int)
    norm_t = torch.tensor([t / T])[:, None, None, None].to(device)
    eps_theta = model(x_t, norm_t)  # predict noise
    beta_t = beta[t]
    alpha_t = alpha[t]
    alpha_bar_t = alpha_bar[t]

    coef1 = 1 / torch.sqrt(alpha_t)
    coef2 = beta_t / torch.sqrt(1 - alpha_bar_t)

    mu = coef1 * (x_t - coef2 * eps_theta)
    sigma = torch.sqrt(beta_t)

    noise = torch.randn_like(x_t) if t > 0 else 0
    x_prev = mu + sigma * noise
    return x_prev


def ddim_sample_step(x_t, t, t_prev, model, T):
    norm_t = torch.tensor([t / T])[:, None, None, None].to(device)
    eps_theta = model(x_t, norm_t)

    alpha_bar_t = alpha_bar[t]
    alpha_bar_prev = alpha_bar[t_prev]

    x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_bar_t)
    x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev) * eps_theta
    return x_prev


def sample_ddpm(model, shape, T):
    intermediate = []
    x_t = torch.randn(shape, device=device)
    for t in reversed(range(T)):
        x_t = ddpm_sample_step(x_t, t, model, T)
        if t < 50:
            # Save intermediate samples
            intermediate.append(x_t.detach().cpu().numpy())
    return np.stack(intermediate)


def sample_ddim(model, shape, T, ddim_steps=50):
    x_t = torch.randn(shape, device=device)
    timesteps = torch.linspace(0, T - 1, ddim_steps, dtype=torch.long)
    intermediate = []
    for i in reversed(range(1, len(timesteps))):
        t = timesteps[i]
        t_prev = timesteps[i - 1]
        x_t = ddim_sample_step(x_t, t, t_prev, model, T)
        if t < 50:
            # Save intermediate samples
            intermediate.append(x_t.detach().cpu().numpy())
    return np.stack(intermediate)

n_sample = 8

with torch.no_grad():
    ddpm_intermediate = sample_ddpm(
        model.eval(),
        shape=(n_sample, 3, pixel_height, pixel_height),
        T=time_steps,
        save_rate=save_rate
    )

with torch.no_grad():
    ddim_intermediate = sample_ddim(
        model.eval(),
        shape=(n_sample, 3, pixel_height, pixel_height),
        T=time_steps,
        save_rate=save_rate
    )


plot = plot_sample(
    ddpm_intermediate,
    n_sample,
    4,
    save_dir,
    "ddpm_intermediate",
    None,
    save=True
)

plot = plot_sample(
    ddim_intermediate,
    n_sample,
    4,
    save_dir,
    "ddim_intermediate",
    None,
    save=True
)