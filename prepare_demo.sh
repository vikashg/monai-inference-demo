# Install gdown 
#pip install --upgrade --no-cache-dir gdown

# Download large files and example data 
#gdown --folder https://drive.google.com/drive/folders/1Y863tgKMdmB_E_Y0fF5MrH5qdK5Hqp0B?usp=sharing

# Copy files to appropriate folders 
mkdir -p TritonInferenceServer/data
cp -v MONAI-model-inference/data_store/breast_density/breast.jpg TritonInferenceServer/data/breast.jpg
cp -v MONAI-model-inference/data_store/lv_segmentation/test.nii.gz TritonInferenceServer/data/test.nii.gz
cp -rv MONAI-model-inference/model_repository TritonInferenceServer
 

