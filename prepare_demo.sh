# Install gdown 
#pip install --upgrade --no-cache-dir gdown

# Download large files and example data 
#gdown --folder https://drive.google.com/drive/folders/1Y863tgKMdmB_E_Y0fF5MrH5qdK5Hqp0B?usp=sharing

# Copy files to appropriate folders 
mkdir -p TritonInferenceServer/data
cp -v MONAI-model-inference/data_store/breast_density/breast.jpg TritonInferenceServer/data/breast.jpg
cp -v MONAI-model-inference/data_store/lv_segmentation/test.nii.gz TritonInferenceServer/data/test.nii.gz
cp -rv MONAI-model-inference/model_repository TritonInferenceServer
 
# Copy file for torchserve
cp -rv MONAI-model-inferece/data_store torchserve/
cp -rv MONAI-model-inferece/model-store torchserve/model-store


# Copy models to individual model testing 
mkdir -p torchserve/breast_density_classification/model_store
cp -v MONAI-model-inferece/model-store/breast.mar torchserve/breast_densty_classification/model_store/breast.mar

mkdir -p torchserve/lv_segmentation/model_store
cp -v MONAI-model-inference/model-store/segresnet.mar torchserve/lv_segmentation/model_store/segresnet.mar

# Copy data for individual testing
# copy lv_segmentation data
cp -v MONAI-model-inference/data_store/lv_segmentation/test.nii.gz torchserve/lv_segmentation/test.nii.gz
# copt breast_density segmentation data 
cp -v MONAI-model-inference/data_store/breast_density/breast.jpg torchserve/breast_density/breast.jpg
