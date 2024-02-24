import numpy as np 
import tritonclient.http as httpclient
from PIL import Image 
from monai.transforms import LoadImage
from tritonclient.utils import triton_to_np_dtype
from monai.transforms import (Compose, LoadImage, EnsureChannelFirst,
														ScaleIntensity, Resize)


def preprocess(img_path="../data/breast.jpg"):

	transforms = Compose([LoadImage(image_only=True), 
												EnsureChannelFirst(channel_dim=2),
												ScaleIntensity(minv=0.0, maxv=1.0),
												Resize(spatial_size=(299, 299))
												])

	results_np = np.expand_dims(transforms(img_path).numpy(), axis=0)
	print(results_np.shape)

	return results_np


def main():
	transformed_image = preprocess()
	client = httpclient.InferenceServerClient(url="localhost:8000")
	
	inputs = httpclient.InferInput("input__0", transformed_image.shape, datatype="FP32")
	inputs.set_data_from_numpy(transformed_image, binary_data=True)

	outputs = httpclient.InferRequestedOutput("output__0", binary_data=True, class_count =0)
	results = client.infer(model_name="breast_density", inputs=[inputs], outputs=[outputs])
	inference_output = results.as_numpy("output__0")
	print(inference_output)

	

if __name__ == '__main__':
	main()
