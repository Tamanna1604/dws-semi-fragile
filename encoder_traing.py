# watermark_training.py
"""
End-to-end PyTorch training script for selective-robust watermarking:
- Encoder (U-Net) embeds a bit-string message into an image
- Decoder recovers message from transformed images
- Patch discriminator to encourage visual realism
- Benign and malicious transforms applied during training
"""

import os
import math
import random
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils
import kornia.augmentation as K
import kornia.geometry.transform as KT
import lpips  # perceptual loss
# Note: if lpips import fails, pip install lpips

# ---------------------------
# Utilities
# ---------------------------

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias.data, 0.0)

# ---------------------------
# U-Net blocks (simple)
# ---------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # use conv + nearest upsample (avoid transpose conv artifacts)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if different sizes
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# ---------------------------
# Encoder (U-Net style) - outputs watermarked RGB image
# ---------------------------
class UNetEncoder(nn.Module):
    def __init__(self, in_channels=4, base_ch=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.down4 = Down(base_ch*8, base_ch*8)
        # bottleneck -> up
        self.up1 = Up(base_ch*8, base_ch*8)
        self.up2 = Up(base_ch*8, base_ch*4)
        self.up3 = Up(base_ch*4, base_ch*2)
        self.up4 = Up(base_ch*2, base_ch)
        # output conv to RGB
        self.outc = nn.Conv2d(base_ch, 3, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = torch.tanh(self.outc(x))  # range [-1,1]
        return out

# ---------------------------
# Decoder (U-Net style) - outputs L-length vector (sigmoid)
# ---------------------------
class UNetDecoder(nn.Module):
    def __init__(self, in_channels=3, base_ch=32, msg_len=128):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.down4 = Down(base_ch*8, base_ch*8)
        self.up1 = Up(base_ch*8, base_ch*8)
        self.up2 = Up(base_ch*8, base_ch*4)
        self.up3 = Up(base_ch*4, base_ch*2)
        self.up4 = Up(base_ch*2, base_ch)
        # produce a spatial message projection (e.g., 96x96 -> downsample to message)
        self.final_conv = nn.Conv2d(base_ch, 1, kernel_size=1)
        self.msg_len = msg_len
        # linear projection to message
        self.fc = nn.Linear(96*96, msg_len)
    def forward(self, x):
        # x expected in [-1,1] RGB
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_ = self.up1(x5, x4)
        x_ = self.up2(x_, x3)
        x_ = self.up3(x_, x2)
        x_ = self.up4(x_, x1)
        # produce spatial map
        sp = self.final_conv(x_)  # (B,1,H,W)
        # resize to 96x96 then flatten
        sp96 = F.adaptive_avg_pool2d(sp, (96,96)).view(sp.size(0), -1)
        vec = self.fc(sp96)
        return torch.sigmoid(vec)  # each bit in [0,1]

# ---------------------------
# Patch Discriminator (simple)
# ---------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1),  # 128 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch*2, 4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(base_ch*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch*2, base_ch*4, 4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(base_ch*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch*4, 1, 4, padding=1)  # patch response map
        )
    def forward(self, x):
        # returns patch logits (B,1,H',W') -> average to scalar probability
        return torch.sigmoid(self.net(x)).view(x.size(0), -1).mean(dim=1)  # (B,)

# ---------------------------
# Message helper: projection and injection
# ---------------------------
class MessageProjector(nn.Module):
    """
    Project binary message vector (L,) to a spatial map, then resize to image resolution
    Used before injecting as 4th channel.
    """
    def __init__(self, msg_len=128, proj_size=96, out_res=256):
        super().__init__()
        self.msg_len = msg_len
        self.proj_size = proj_size
        self.out_res = out_res
        self.fc = nn.Linear(msg_len, proj_size*proj_size)
    def forward(self, s):
        # s: (B, L) float in {0,1}
        B = s.size(0)
        sp = self.fc(s)  # (B,proj_size*proj_size)
        sp = sp.view(B, 1, self.proj_size, self.proj_size)
        # bilinear upsample to out_res
        sp_up = F.interpolate(sp, size=(self.out_res, self.out_res), mode='bilinear', align_corners=False)
        return sp_up  # (B,1,H,W)

# ---------------------------
# Differentiable transforms (benign) using Kornia
# ---------------------------
class BenignTransform(nn.Module):
    def __init__(self, out_size=256):
        super().__init__()
        # We'll use kornia augmentations; each is differentiable
        self.aug = nn.Sequential(
            # random Gaussian blur (kernel size random)
            K.RandomGaussianBlur((3,7), (0.1,2.0), p=0.5),
            # random brightness/contrast/saturation
            K.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.3, p=0.8),
            # small rotation/translation
            K.RandomAffine(degrees=10.0, translate=(0.04,0.04), p=0.6),
            # random resized scale (down-up)
            K.RandomResizedCrop((out_size, out_size), scale=(1/5.0, 1.0), ratio=(1.0,1.0), p=0.6)
        )
        # Note: differentiable JPEG is not in kornia directly; you can implement or import a known differentiable JPEG.
        # For this script, we will simulate via small quantization smoothing (approx).
    def forward(self, x):
        # x range [-1,1] --> kornia expects [0,1]
        x01 = (x + 1.0) / 2.0
        out = self.aug(x01)
        # optionally apply JPEG-like quantization smoothing (approx)
        return out*2.0 - 1.0

# ---------------------------
# Malicious transform: placeholder heavy edits (non-diff)
# ---------------------------
class MaliciousTransform:
    """
    For malicious transforms we often use non-differentiable heavy edits:
    - face swap outputs (precomputed)
    - strong GAN-based manipulations
    - heavy smoothing + color distortions + cropping/rescaling
    This class is a placeholder: it may be non-differentiable; we'll treat it as a "stop_grad" branch.
    """
    def __init__(self):
        # you can replace this with calls to a pre-trained face-swap model to synthesize xm
        pass
    def __call__(self, x):
        # x: torch Tensor (B,3,H,W) in [-1,1]
        # Simple strong edit approximation: heavy JPEG (non-diff) + large color change + smoothing
        # Move to CPU and use PIL for non-differentiable ops (simulate real-world edits)
        x_cpu = (x.detach().cpu() + 1.0) * 127.5  # [0,255]
        x_cpu = x_cpu.permute(0,2,3,1).numpy().astype('uint8')  # (B,H,W,3)
        out_list = []
        from PIL import Image, ImageFilter, ImageEnhance
        for im_np in x_cpu:
            im = Image.fromarray(im_np)
            # strong blur
            im = im.filter(ImageFilter.GaussianBlur(radius=3))
            # aggressive contrast
            im = ImageEnhance.Contrast(im).enhance(1.8)
            # heavy JPEG: save and reload with low quality
            import io
            buf = io.BytesIO()
            im.save(buf, format='JPEG', quality=40)
            buf.seek(0)
            im = Image.open(buf).convert('RGB')
            out_list.append(torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes())).view(im.size[1], im.size[0], 3).numpy())).permute(2,0,1))
            # NOTE: above is clunky; better to use numpy.asarray(im) if pillow available
        # Simpler: just convert via numpy (safer)
        import numpy as np
        out_tensors = []
        for im in out_list:
            # expecting (3,H,W) uint8
            imf = im.float()  # already torch
            out_tensors.append(imf)
        if len(out_tensors) == 0:
            # fallback: return detached input (no change)
            return x.detach()
        bat = torch.stack(out_tensors, dim=0).to(x.device)  # (B,3,H,W)
        bat = (bat / 127.5) - 1.0
        return bat

# ---------------------------
# Loss objects and training step
# ---------------------------
class WatermarkTrainer:
    def __init__(self, cfg):
        self.device = cfg['device']
        # models
        self.encoder = UNetEncoder(in_channels=4, base_ch=32).to(self.device)
        self.decoder = UNetDecoder(in_channels=3, base_ch=32, msg_len=cfg['msg_len']).to(self.device)
        self.discriminator = PatchDiscriminator(in_ch=3, base_ch=64).to(self.device)
        self.msg_proj = MessageProjector(msg_len=cfg['msg_len'], proj_size=96, out_res=cfg['img_size']).to(self.device)
        # losses
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)  # LPIPS; expects [-1,1]
        # optimizers
        gen_params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.msg_proj.parameters())
        self.opt_gen = torch.optim.Adam(gen_params, lr=cfg['lr'], betas=(0.5,0.999))
        self.opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=cfg['lr']*0.5, betas=(0.5,0.999))
        # transforms
        self.benign = BenignTransform(out_size=cfg['img_size']).to(self.device)
        self.malicious = MaliciousTransform()  # possibly non-diff
        # hyperparams
        self.cp = cfg.get('cp', 1.0)
        self.cg = cfg.get('cg', 1e-2)
        self.cM = cfg.get('cM', 1.0)
        self.msg_len = cfg['msg_len']

    def train_epoch(self, dataloader):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(self.device)  # expect in [-1,1]
            B = imgs.size(0)
            # sample random binary messages
            s = torch.randint(0,2,(B, self.msg_len)).float().to(self.device)  # {0,1}
            # project message -> spatial map and concat as 4th channel
            s_map = self.msg_proj(s)  # (B,1,H,W)
            enc_in = torch.cat([imgs, s_map], dim=1)  # (B,4,H,W)
            # generate watermarked image
            xw = self.encoder(enc_in)  # [-1,1]
            # discriminator update
            # real label: images, fake: xw
            self.opt_disc.zero_grad()
            pred_real = self.discriminator(imgs)  # (B,)
            pred_fake = self.discriminator(xw.detach())
            # BCE loss: real->1, fake->0
            loss_d_real = F.binary_cross_entropy(pred_real, torch.ones_like(pred_real))
            loss_d_fake = F.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake))
            loss_disc = (loss_d_real + loss_d_fake) * 0.5
            loss_disc.backward()
            self.opt_disc.step()

            # generator + decoder update
            self.opt_gen.zero_grad()
            # benign transformed images (differentiable)
            xb = self.benign(xw)  # differentiable chain
            # malicious transformed images (may be non-diff)
            xm = self.malicious(xw)  # treat as detached potentially
            # predictions
            sb = self.decoder(xb)
            sm = self.decoder(xm.detach() if xm.requires_grad == False else xm)  # if non-diff, detach
            # message losses
            # L1 between ground truth bits and predicted bits (use sigmoid outputs in [0,1])
            loss_msg_b = self.l1(sb, s)
            loss_msg_m = self.l1(sm, s)
            L_M = loss_msg_b - loss_msg_m  # we want minimize this (low on benign, high on malicious)
            # image reconstruction losses
            loss_l1_img = self.l1(xw, imgs)
            loss_l2_img = self.l2(xw, imgs)
            loss_lpips = self.lpips_fn((xw+1)/2.0, (imgs+1)/2.0).mean()  # lpips expects [0,1] or normalized? verify docs
            L_d = loss_l1_img + loss_l2_img + self.cp * loss_lpips
            # adversarial generator loss: encourage discriminator to label xw as real (1)
            pred_fake_forG = self.discriminator(xw)
            loss_Gadv = F.binary_cross_entropy(pred_fake_forG, torch.ones_like(pred_fake_forG))
            L_img = L_d + self.cg * loss_Gadv
            # combined
            total_gen_loss = L_img + self.cM * L_M
            total_gen_loss.backward()
            self.opt_gen.step()

            if i % 50 == 0:
                # print training stats
                print(f"Iter {i}: D_loss={loss_disc.item():.4f} Gen_loss={total_gen_loss.item():.4f} Lmsg_b={loss_msg_b.item():.4f} Lmsg_m={loss_msg_m.item():.4f} Llpips={loss_lpips.item():.4f}")

# ---------------------------
# Example usage with dataset
# ---------------------------
def build_dataloader(img_size=256, batch_size=8, root='./data/imagenet_subset'):
    # You can use ImageFolder with transforms: Resize to img_size and normalize to [-1,1]
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])  # map to [-1,1]
    ])
    dataset = datasets.ImageFolder(root=root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return loader

if __name__ == '__main__':
    cfg = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'img_size': 256,
        'msg_len': 128,
        'lr': 2e-4,
        'cp': 1.0,
        'cg': 1e-2,
        'cM': 1.0
    }
    loader = build_dataloader(img_size=cfg['img_size'], batch_size=8, root='path_to_your_images')
    trainer = WatermarkTrainer(cfg)
    for epoch in range(1, 51):
        print("Epoch", epoch)
        trainer.train_epoch(loader)
        # optionally save models
        torch.save({
            'encoder': trainer.encoder.state_dict(),
            'decoder': trainer.decoder.state_dict(),
            'discriminator': trainer.discriminator.state_dict()
        }, f'checkpoint_epoch_{epoch}.pth')
