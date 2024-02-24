docker run --gpus=1 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository/models:/models nvcr.io/nvidia/tritonserver:23.12-py3 tritonserver --model-repository=/models

