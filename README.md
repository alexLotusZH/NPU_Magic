# NPU_Magic

## Description
A Project of ISC Lab, Umich. Convert yolov8 model into a quantized one that runs on RK3588 for pose detection.
Able to do pose detection on a single image or a live-stream video.

## Usage
1. Convert to .rknn model
python3 convert.py [onnx_model_path] [platform] [dtype(optional)] [output_rknn_path(optional)]

platform choose from [rk3562,rk3566,rk3568,rk3588]

dtype choose from    [i8, fp]

The size of the input should be (224,224) under i8(quantized mode)

example: cd into ./src and run python convert.py ../model/yolov8s-pose.onnx rk3588 fp

## Dependency library installation
Requirements and wheel for rknntoolkit1.6.0 are provided in the repo.

## Model performance benchmark(FPS)
About 20fps using a 60hz live cam in live demo. 

For NPU performance: please refer to this note: https://www.notion.so/NPU-ML-On-Orange-pi-3f66c06106e945549b0cfa740b25fe6a

## Credits
This project is based on the `RKNN Model Zoo` (https://github.com/airockchip/rknn_model_zoo)


