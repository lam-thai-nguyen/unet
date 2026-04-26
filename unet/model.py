import torch
import torch.nn as nn


class UNet(nn.Module):

	def __init__(self):
		super().__init__()

		self.contracting = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

			nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
		)

		self.bottleneck = nn.Sequential(
			nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(1024),
			nn.ReLU(inplace=True),
			nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(1024),
			nn.ReLU(inplace=True),
		)

		self.expansive = nn.Sequential(
			nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
			nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
			nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
			nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
			nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0),
		)

	def forward(self, x):
		crop1, crop2, crop3, crop4 = None, None, None, None
		for name, layer in self.contracting.named_children():
			x = layer(x)
			if name == '5':
				crop1 = x
			elif name == '12':
				crop2 = x
			elif name == '19':
				crop3 = x
			elif name == '26':
				crop4 = x

		x = self.bottleneck(x)

		for name, layer in self.expansive.named_children():
			x = layer(x)
			if name == '0':
				x = torch.cat((crop4,x), dim=1)  # type: ignore[call-overload]
			elif name == '7':
				x = torch.cat((crop3,x), dim=1)  # type: ignore[call-overload]
			elif name == '14':
				x = torch.cat((crop2,x), dim=1)  # type: ignore[call-overload]
			elif name == '21':
				x = torch.cat((crop1,x), dim=1)  # type: ignore[call-overload]

		return x