from ts.torch_handler.base_handler import BaseHandler
from monai.transforms import (Compose, 
					EnsureChannelFirstd,
					LoadImaged, AddChanneld, 
					Activations,
					AsDiscreted, 
					Resized,
					SaveImaged,
					ScaleIntensityd,
					ScaleIntensityRanged)
import torch
import logging
import json
import SimpleITK as sitk 
import numpy as np 
import os

input_dir = '../data_store/breast_density'

class ModelHandler(BaseHandler):
	def __init__(self):
		self._context = None
		self.initialized = False
		self.explain = False
		self.target = 0 
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def inference(self, data, *args, **kwargs):
		with torch.no_grad():
			marshalled_data = data['image']['image'].to(self.device)
			results = self.model(marshalled_data, *args, *kwargs)
		return results

	def preprocess(self, data_fn):
		transforms = Compose([LoadImaged(keys = 'image' , image_only = True), 
													EnsureChannelFirstd(keys='image', channel_dim=2), 
													Resized( keys='image', spatial_size=(299, 299)),
							  					AddChanneld(keys='image')])

		print('FileName ', data_fn)
		
		fn = data_fn[0]['filename'].decode()
		#fn = data_fn
		img_fullname = os.path.join(input_dir, fn)
		# img_fullname = data_fn
		data = []
		batch_size = 1 
		for i in range(batch_size):
			tmp = {}
			print(img_fullname)
			tmp["data"] = transforms({'image': img_fullname})
			data.append(tmp)
		return data

	def postprocess(self, inference_output):
		print(inference_output)
		post_trans = Activations(sigmoid=True)
		postprocess_output = post_trans(inference_output)
		#convert metatensor to list
		postprocess_output = postprocess_output.tolist()

		result_dict = [{"A": postprocess_output[0][0],
					   						"B": postprocess_output[0][1],
					   						"C": postprocess_output[0][2],
					   						"D": postprocess_output[0][3]
					   	}]
		print("Result Dict: ", result_dict)
		return result_dict

		
	def handle(self, data, context):
		model_input = self.preprocess(data)
		model_output = self.inference({'image':model_input[0]['data']})
		return self.postprocess(model_output)

