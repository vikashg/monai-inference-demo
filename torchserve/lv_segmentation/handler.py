from ts.torch_handler.base_handler import BaseHandler
from monai.transforms import (Compose, EnsureChannelFirst, LoadImage, Activations, AsDiscrete, Resize, AddChannel , SaveImage, ScaleIntensityRange)
import torch
import logging
import json
import SimpleITK as sitk
import numpy as np
import os

data_dir = './data_store/lv_segmentation'
out_dir='./data_store/results_dir'

class ModelHandler(BaseHandler):
	def __init__(self):
		self._context = None
		self.initialized = False
		self.explain = False 
		self.target = 0
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
	
	def inference(self, data, *args, **kwargs):
		with torch.no_grad():
			marshalled_data = data.to(self.device)
			results = self.model(marshalled_data, *args, *kwargs)
		return results

	def preprocess(self, data_fn):
		transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), Resize(spatial_size=(256, 256, 24)), ScaleIntensityRange(a_min=20, a_max=1200, b_min=0, b_max=1,clip=True),AddChannel()])


		print("DATA_FN ", data_fn)
		fn = data_fn[0]['filename'].decode()
		img_fullname = os.path.join(data_dir, fn)
		#img_fullname = data_fn
		reader = sitk.ImageFileReader()
		data = []
		batch_size = 1
		for i in range(batch_size):
			tmp = {}
			reader.SetFileName(img_fullname)
			image = reader.Execute()
			image_np = sitk.GetArrayFromImage(image)

			new_size = [256, 256, 24]
			resample = sitk.ResampleImageFilter()
			resample.SetInterpolator = sitk.sitkLinear
			resample.SetOutputDirection = image.GetDirection()
			resample.SetOutputOrigin = image.GetOrigin()
			resample.SetSize(new_size)
			orig_size = np.array(image.GetSize())
			orig_spacing = np.array(image.GetSpacing())
			new_spacing = orig_size*(orig_spacing/new_size)
			print("New Spacing: ", new_spacing)


			resampled_image = resample.Execute(image)
			resampled_image_np = np.expand_dims(sitk.GetArrayFromImage(resampled_image), axis=0)
			tmp["data"] = transforms(img_fullname)
			data.append(tmp)
		return data

	def postprocess(self,  inference_output):
		print("Inference Output: ", inference_output)
		post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5),])
		trans_save= SaveImage(output_dir = out_dir, output_postfix = "segmentation", output_ext='nii.gz', resample=True) 
		postprocess_output = post_trans(inference_output)
		trans_save(postprocess_output[0,])
		return [1]

	def handle(self, data, context):
		model_input = self.preprocess(data)
		model_output = self.inference(model_input[0]['data'])
		return self.postprocess(model_output)	
