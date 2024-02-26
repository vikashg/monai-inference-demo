import torch
import monai 
from torchviz import make_dot

class ModelDefinition:
	def __init__(self, model_name='unet'):
		self.model = None
		self.model_architecture(model_name = model_name)


	def model_architecture(self, model_name = 'unet'):
		if model_name == 'unet':
			self.model = monai.networks.nets.UNet(
				spatial_dims=3,
				in_channels = 1, 
				out_channels = 1, 
				channels = (16, 32, 64, 128, 256), 
				strides = (2, 2, 2, 2),
				num_res_units=2,
			)

		elif model_name == "BasicUNetPlusPlus":
			self.model = monai.networks.nets.BasicUNetPlusPlus(
				spatial_dims = 3, 
				in_channels=1,
				out_channels=2,
				features=(32, 32, 64, 128, 256, 32), 
				deep_supervision=False, 
				act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}),
				norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv')

		elif model_name == 'SegResNet':
			self.model = monai.networks.nets.SegResNet(
					spatial_dims = 3,
					blocks_down = [1, 2, 2, 4], 
					blocks_up = [1, 1, 1],
					init_filters = 16,
					in_channels=1, 
					out_channels=1,
					dropout_prob=0.2,)

		else:
			print('model name not found')

	def get_model(self):
		return self.model

if __name__ == '__main__':
	model_def = ModelDefinition("SegResNet")
	model = model_def.get_model()
	x = torch.randn(1,1, 256, 256, 256) # For SegResNet
	x = torch.randn(1,1, 128, 128, 16) # For SegResNet
	print(model)
	y = model(x)
	print(y.shape)
	#make_dot(y.mean(), params=dict(model.named_parameters())).render("BasicUNetPlusPlus", format="pdf")

