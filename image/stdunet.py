import torch
import torch.nn as nn
from torch.nn import functional as F


class conv_block(nn.Module):
	def __init__(self, in_c, out_c):
		super().__init__()

		self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(out_c)

		self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(out_c)

		self.relu = nn.ReLU()

	def forward(self, inputs):
		x = self.conv1(inputs)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)

		return x


class encoder_block(nn.Module):
	def __init__(self, in_c, out_c):
		super().__init__()

		self.conv = conv_block(in_c, out_c)
		self.pool = nn.MaxPool2d((2, 2))
		self.pool1 = nn.AvgPool2d((2, 2))
		self.gamma1 = nn.Parameter(torch.zeros(1))
		self.gamma2 = nn.Parameter(torch.zeros(1))

		self.conv1 = nn.Conv2d(out_c * 2, out_c, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
		self.relu = nn.ReLU()

	def forward(self, inputs, std):
		x = self.conv(inputs)

		x = self.gamma1 * std + x

		p_1 = self.pool(x)
		p_2 = self.pool1(x)
		p = torch.cat([p_1, p_2], axis=1)

		std_1 = self.pool(std)
		std_2 = self.pool1(std)
		std = torch.cat([std_1, std_2], axis=1)
		std = self.conv2(std)

		p = self.conv1(p)

		return x, p, std


class decoder_block(nn.Module):
	def __init__(self, in_c, out_c):
		super().__init__()

		self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
		self.conv = conv_block(out_c + out_c, out_c)

	def forward(self, inputs, skip):
		x = self.up(inputs)
		x = torch.cat([x, skip], axis=1)
		x = self.conv(x)
		return x


class build_unet(nn.Module):
	def __init__(self, n_channels, n_classes):
		super().__init__()

		""" Encoder """
		self.e1 = encoder_block(n_channels, 64)
		self.e2 = encoder_block(64, 128)
		self.e3 = encoder_block(128, 256)
		self.e4 = encoder_block(256, 512)

		""" Bottleneck """
		self.b = conv_block(512, 1024)

		""" Decoder """
		self.d1 = decoder_block(1024, 512)
		self.d2 = decoder_block(512, 256)
		self.d3 = decoder_block(256, 128)
		self.d4 = decoder_block(128, 64)

		""" Classifier """
		# self.outputs1 = nn.Conv2d(64, 1, kernel_size=1, padding=0)
		self.outputs = nn.Conv2d(in_channels=64,out_channels=n_classes, kernel_size=1, stride=1,padding=0, bias=True)

		"""可学习超参数"""
		self.gamma1 = nn.Parameter(torch.zeros(1))
		self.gamma2 = nn.Parameter(torch.zeros(1))
		self.gamma3 = nn.Parameter(torch.zeros(1))

	def forward(self, inputs, std):
		""" Encoder """
		std = std ** 2
		std = torch.sigmoid(std)

		# T = 1
		# std = torch.exp(std/T)/(torch.exp(std/T)+torch.exp((1-std)/T))

		s1, p1, stdc = self.e1(inputs, std)

		s2, p2, stdc = self.e2(p1, stdc)

		s3, p3, stdc = self.e3(p2, stdc)

		s4, p4, stdc = self.e4(p3, stdc)

		""" Bottleneck """
		b = self.b(p4)

		""" Decoder """
		d1 = self.d1(b, s4)

		d2 = self.d2(d1, s3)

		d3 = self.d3(d2, s2)

		d4 = self.d4(d3, s1)

		d4 = self.gamma2 * std + d4

		mu = self.outputs(d4)

		return mu


if __name__ == "__main__":
	x = torch.randn((2, 4, 512, 512))
	f = build_unet(n_channels=4, n_classes=1)
	y1, y2 = f(x)
	print(y1.shape)
	print(y2.shape)
