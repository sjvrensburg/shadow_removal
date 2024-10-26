import math

import torch
import torch.nn.functional as F
from torch import nn
from timm.layers import DropPath, to_2tuple, trunc_normal_


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.eval()
    
    @property
    def device(self):
        return next(self.parameters()).device

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class ShiftedMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features  # Do not multiply by mlp_ratio here
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ShiftedBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=1., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ShiftedMLP(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                             padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class UNextGCNet(BaseModel):
    """GCNet model that exactly matches the pretrained weights"""
    def __init__(self, num_classes, input_channels=3, img_size=512):
        super().__init__()
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_channels, 16, 3, stride=1, padding=0)
        )
        self.encoder2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 32, 3, stride=1, padding=0)
        )
        self.encoder3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 128, 3, stride=1, padding=0)
        )
        
        # Batch Norms
        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)
        
        # Transformer blocks
        self.norm3 = nn.LayerNorm(160)
        self.norm4 = nn.LayerNorm(256)
        self.dnorm3 = nn.LayerNorm(160)
        self.dnorm4 = nn.LayerNorm(128)
        
        # Patch embeddings
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                             in_chans=128, embed_dim=160)
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                             in_chans=160, embed_dim=256)
        
        # Shifted blocks
        self.block1 = nn.ModuleList([ShiftedBlock(dim=160)])
        self.block2 = nn.ModuleList([ShiftedBlock(dim=256)])
        self.dblock1 = nn.ModuleList([ShiftedBlock(dim=160)])
        self.dblock2 = nn.ModuleList([ShiftedBlock(dim=128)])
        
        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 160, 3, stride=1, padding=0)
        )
        self.decoder2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(160, 128, 3, stride=1, padding=0)
        )
        self.decoder3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 32, 3, stride=1, padding=0)
        )
        self.decoder4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 16, 3, stride=1, padding=0)
        )
        self.decoder5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 16, 3, stride=1, padding=0)
        )
        
        # Decoder Batch Norms
        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)
        
        # Final layer
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B = x.shape[0]
        
        # Encoder
        temp = self.ebn1(self.encoder1(x))
        out = F.relu(F.max_pool2d(temp, 2, 2))
        t1 = out
        
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out
        
        # Transformer blocks
        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        t4 = out
        
        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        # Decoder
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        
        for blk in self.dblock1:
            out = blk(out, H, W)
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t3)
        
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)
        
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode='bilinear'))
        out = torch.add(out, F.relu(temp))
        
        return self.sigmoid(self.final(out))

class UNextDRNet(BaseModel):
    """Second model (DRNET) with larger dimensions"""
    def __init__(self, num_classes, input_channels=6, img_size=512):
        super().__init__()
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_channels, 32, 3, stride=1, padding=0)
        )
        self.encoder2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3, stride=1, padding=0)
        )
        self.encoder3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, stride=1, padding=0)
        )
        
        # Batch Norms
        self.ebn1 = nn.BatchNorm2d(32)
        self.ebn2 = nn.BatchNorm2d(64)
        self.ebn3 = nn.BatchNorm2d(128)
        
        # Transformer blocks
        self.norm3 = nn.LayerNorm(256)
        self.norm4 = nn.LayerNorm(512)
        self.dnorm3 = nn.LayerNorm(256)
        self.dnorm4 = nn.LayerNorm(128)
        
        # Patch embeddings
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                             in_chans=128, embed_dim=256)
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                             in_chans=256, embed_dim=512)
        
        # Shifted blocks with larger dimensions
        self.block1 = nn.ModuleList([ShiftedBlock(dim=256)])
        self.block2 = nn.ModuleList([ShiftedBlock(dim=512)])
        self.dblock1 = nn.ModuleList([ShiftedBlock(dim=256)])
        self.dblock2 = nn.ModuleList([ShiftedBlock(dim=128)])
        
        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3, stride=1, padding=0)
        )
        self.decoder2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3, stride=1, padding=0)
        )
        self.decoder3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3, stride=1, padding=0)
        )
        self.decoder4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, 3, stride=1, padding=0)
        )
        self.decoder5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, 3, stride=1, padding=0)
        )
        
        # Decoder Batch Norms
        self.dbn1 = nn.BatchNorm2d(256)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(64)
        self.dbn4 = nn.BatchNorm2d(32)
        self.dbn5 = nn.BatchNorm2d(32)
        
        # Additional outputs
        self.out8 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=1)
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1)
        )
        self.out2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1)
        )
        
        # Final layer
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B = x.shape[0]
        
        # Encoder
        temp = self.ebn1(self.encoder1(x))
        out = F.relu(F.max_pool2d(temp, 2, 2))
        t1 = out
        
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out
        
        # Transformer blocks
        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        t4 = out
        
        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        # Decoder
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        
        for blk in self.dblock1:
            out = blk(out, H, W)
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t3)
        
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)
        
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        # Additional outputs
        out8 = self.sigmoid(self.out8(out))
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t2)
        
        out4 = self.sigmoid(self.out4(out))
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t1)
        
        out2 = self.sigmoid(self.out2(out))
        out = F.relu(F.interpolate(self.dbn5(self.decoder5(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, F.relu(temp))
        
        return self.sigmoid(self.final(out)), out2, out4, out8
    