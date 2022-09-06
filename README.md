# ainference
aiosync inference benchmarking


Using async code to run inference the whole imagenet validation dataset. Backend server is torchserve using image classification handler.
Asynchronous requests are ideal for batch processing on the torchserve side.
