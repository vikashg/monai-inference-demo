torch-model-archiver --model-name breast --serialized-file ./traced_ts_model.pt --handler ./handler.py -v 1.0
mv breast.mar model_store

torchserve --start --model-store model_store/ --models breast=breast.mar
curl -d "filename=breast.jpg" http://127.0.0.1:8080/predictions/breast
