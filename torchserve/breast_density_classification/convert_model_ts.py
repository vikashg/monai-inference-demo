import torch
import torchviz
from monai.networks.nets.torchvision_fc import TorchVisionFCModel 
from torchvision.models.inception import Inception_V3_Weights

'''
model = TorchVisionFCModel("inception_v3", num_classes=4, in_channels=1,
								use_conv=False, pool=None)
model_fn = '/home/gupta/disk/Tools/Vikash/LVSegmentation/monai-models/breast_density_classification/models/model.pt'
model.load_state_dict(torch.load(model_fn))

x = torch.zeros(1, 1, 299, 299)
traced_model = torch.jit.trace(model, x)
traced_model.save('./traced_inception_model.pt')
'''
from torchvision import models 
import torch.nn as nn

model = models.inception_v3(pretrained=1)
model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 4))
model.aux_logits = False

model_fn = '/home/gupta/disk/Tools/Vikash/LVSegmentation/monai-models/breast_density_classification/models/model.pt'
model.load_state_dict(torch.load(model_fn))
x = torch.zeros(1, 299, 299)
traced_model = torch.jit.trace(model, x)
traced_model.save('/home/gupta/disk/Tools/Vikash/LVSegmentation/monai-models/breast_density_classification/torch-serve/traced_inception_model.pt')
