# Create a model archive from the torch-script model
torch-model-archiver --model-name lvsegmentation --serialized-file ./traced_segres_model.pt --handler ./handler.py -v 1.0

# Copy the model to the model-store
mv segresnet.mar model_store

# Start torch serve
torchserve --start --model-store model_store/ --models segresnet=segresnet.mar

# Make a restful call for model prediction
curl  -d "filename=test.nii.gz" http://127.0.0.1:8080/predictions/segresnet
