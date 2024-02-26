FROM pytorch/torchserve:latest
RUN pip install monai 
RUN pip install SimpleITK
RUN pip install torch
COPY model-store /home/model-server/model-store/
CMD ["torchserve" , "--start", "--model-store", "/home/model-server/model-store/", "--models", "all"]
