import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
import tempfile
import os

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_labels=2, img_h=64, img_w=64):
        super().__init__()
        self.enc1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottom = DoubleConv(256, 512)

        def create_decoder_path():
            return nn.ModuleDict({
                'upconv1': nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                'conv1': DoubleConv(512, 256),
                'upconv2': nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                'conv2': DoubleConv(256, 128),
                'upconv3': nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                'conv3': DoubleConv(128, 64),
                'out': nn.Conv2d(64, n_labels, kernel_size=1)
            })

        self.decoder1 = create_decoder_path()
        self.decoder2 = create_decoder_path()
        self.decoder3 = create_decoder_path()

    def _decoder_forward(self, x7, x5, x3, x1, decoder):
        u1 = decoder['upconv1'](x7)
        u1 = torch.cat([u1, x5], dim=1)
        u1 = decoder['conv1'](u1)

        u2 = decoder['upconv2'](u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = decoder['conv2'](u2)

        u3 = decoder['upconv3'](u2)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = decoder['conv3'](u3)

        return decoder['out'](u3)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool1(x1)
        x3 = self.enc2(x2)
        x4 = self.pool2(x3)
        x5 = self.enc3(x4)
        x6 = self.pool3(x5)
        x7 = self.bottom(x6)

        out1 = self._decoder_forward(x7, x5, x3, x1, self.decoder1)
        out2 = self._decoder_forward(x7, x5, x3, x1, self.decoder2)
        out3 = self._decoder_forward(x7, x5, x3, x1, self.decoder3)

        return [out1, out2, out3]


@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    weights_path = hf_hub_download(
        repo_id="vinster619/UNet_USC_TIMIT",
        filename="best_unet_model.pth"
    )
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device
