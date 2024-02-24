import numpy as np
import tritonclient.http as httpclient
import os
from PIL import Image
from monai.transforms import LoadImage
from tritonclient.utils import triton_to_np_dtype
from monai.transforms import (Compose, LoadImage, EnsureChannelFirst,
                                Activations, AsDiscrete, SaveImage,
                              ScaleIntensityRange, Resize)
import SimpleITK as sitk


def preprocess(img_path="../data/test.nii.gz"):
    transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(),
                          Resize(spatial_size=(256, 256, 24)),
                          ScaleIntensityRange(a_min=20, a_max=1200, b_min=0, b_max=1, clip=True)])

    tmp_folder = './tmp'
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    _trans_list = Compose([LoadImage(image_only=True), EnsureChannelFirst(), Resize(spatial_size=(256, 256, 24)),])
    a = _trans_list(img_path)
    saver = SaveImage(output_dir=tmp_folder, output_postfix="preprocessed", output_ext='nii.gz', resample=False)
    saver(a)

    img_tensor = transforms(img_path)
    results_np = np.expand_dims(img_tensor.numpy(), axis=0)
    return results_np


def post_transform(inference_output, out_dir='./', ref_image=None):
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5), ])

    trans_save = SaveImage(output_dir=out_dir, output_postfix="segmentation", output_ext='nii.gz', resample=True)
    postprocess_output = post_trans(inference_output)[0,0,]

    image_itk = sitk.GetImageFromArray(np.transpose(postprocess_output, [2, 1, 0]))
    image_itk.SetSpacing(ref_image.GetSpacing())
    image_itk.SetOrigin(ref_image.GetOrigin())
    image_itk.SetDirection(ref_image.GetDirection())
    sitk.WriteImage(image_itk, os.path.join(out_dir, "segmentation.nii.gz"))


    #print("Inference Output: ", postprocess_output[0,0,].shape)
    #trans_save(postprocess_output[0,])


def main():
    img_path = "../data/test.nii.gz"
    transformed_image = preprocess(img_path=img_path)
    client = httpclient.InferenceServerClient(url="localhost:8000")
    inputs = httpclient.InferInput("input__0", transformed_image.shape, datatype="FP32")
    inputs.set_data_from_numpy(transformed_image, binary_data=True)
    outputs = httpclient.InferRequestedOutput("output__0", binary_data=True, class_count=0)

    results = client.infer(model_name="lv_segmentation", inputs=[inputs], outputs=[outputs])
    inference_output = results.as_numpy("output__0")

    ref_img_fn = './tmp/MR/MR_preprocessed.nii.gz'

    reader = sitk.ImageFileReader()
    reader.SetFileName(ref_img_fn)
    ref_image = reader.Execute()

    post_transform(inference_output, out_dir='./', ref_image = ref_image)


if __name__ == '__main__':
    main()
