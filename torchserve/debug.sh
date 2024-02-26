# Run torchserve
#Author: Vikash Gupta
#Last Updated: 2/26/2024
# change the data_store to your file location

data_dir=/home/gupta/disk/Tools/Vikash/monai-models/data_store
docker build -t torchserve:0.1.0 .
docker run -dp 8080:8080 -p 8081:8081 -v $data_dir:/home/data_store --name torchserve torchserve:0.1.0
