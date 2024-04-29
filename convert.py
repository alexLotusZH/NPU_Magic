import os
import sys
import cv2
import numpy as np
from rknn.api import RKNN
from ultralytics.utils.ops import scale_coords

DATASET_PATH = '../../../datasets/COCO/coco_subset_20.txt'
DEFAULT_RKNN_PATH = '../model/yolov8s-pose.rknn'
DEFAULT_QUANT = True

realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

from py_utils.pose_result_interpretor import Pose_output
from py_utils.coco_utils import COCO_test_helper


def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)]".format(sys.argv[0]))
        print("       platform choose from [rk3562,rk3566,rk3568,rk3588]")
        print("       dtype choose from    [i8, fp]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['i8', 'fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type == 'i8':
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = DEFAULT_RKNN_PATH

    return model_path, platform, do_quant, output_path  


if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()
    img_path = './bus.jpg'

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[
                    [255, 255, 255]], target_platform=platform)#, quantized_dtype="asymmetric_quantized-8")
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    # ret = rknn.load_pytorch(model=model_path, input_size_list = [[1,3,640,640]])

    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    co_helper = COCO_test_helper(enable_letter_box=True)

    # Set inputs
    size = (224,224)
    img = cv2.imread(img_path)
    # img = co_helper.letter_box(im= img, new_shape=size, pad_color=(114, 114, 114))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)


    print('--> Init runtime environment')
    #ret = rknn.init_runtime()
    ret = rknn.init_runtime(target='rk3588', device_id='c1711783a9543aa6', perf_debug = False)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')  
    outputs = rknn.inference(inputs=[img])
    print(outputs)
    print('done')
    pose_result = Pose_output(outputs,img_path,4,size=size)
    pose_result.print_top_keypoints()

    # Release
    perf_detail = rknn.eval_perf(is_print=True)

    rknn.release() 


