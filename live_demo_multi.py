import cv2
import sys
import os
import time
import numpy as np
from rknn.api import RKNN
from multiprocessing import Process, shared_memory, Event, Lock




# capture process definition
def capture_frames(frame_buffer, stop_event, lock):
    

    cap = cv2.VideoCapture(6)
    # cap.set(cv2.CAP_PROP_FPS, 60)

    #codec = 0x47504A4D # MJPG
    #codec = 844715353.0 # YUY2
    # codec = 1196444237.0 # MJPG
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    # #print 'fourcc:', decode_fourcc(codec)
    # cap.set(cv2.CAP_PROP_FOURCC, codec)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60.0)


    while not stop_event.is_set():
        ret, frame = cap.read()
        #print("shape:",frame.shape)
        # print("capture processing")
        # prescaling on this returned frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # write this frame to the memmap "frame_buffer"
        #print(np.max(frame))
        # cv2.imshow('Live demo', frame)

        with lock:
            frame_buffer[:,:,:] = frame.copy()
            # cv2.imshow('Live demo', frame_buffer)
            # print("cap:", frame_buffer)
            #frame_buffer.flush()
            #print("cap:", frame_buffer)
        

        # keyboard listen 
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     stop_event.set() 
        #     break

    # frame_buffer = np.memmap(memmap_path, dtype=dtype, mode='r+', shape=shape)
    # print(frame_buffer)
    cap.release()
    #cap.close()
    cv2.destroyAllWindows()

def process_frames(frame_buffer, stop_event, lock):

    # if stop_event.is_set():
    #     print("Stop event is set")
    # else:
    #     print("Stop event is not set")

    realpath = os.path.abspath(__file__)
    _sep = os.path.sep
    realpath = realpath.split(_sep)
    sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

    from py_utils.pose_result_interpretor import transform_frame, plot_keypoints
    from py_utils.coco_utils import COCO_test_helper

    # setup phase
    model_path = "/home/puffin/Desktop/rknn_model_zoo/examples/yolov8/model/yolov8s-pose.rknn"

    rknn = RKNN(verbose=False)

    print('--> Load model')
    ret = rknn.load_rknn(path=model_path)
    if ret != 0:
        print("Load rknn model failed")
        exit(ret)
    print('done')

    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk3588', device_id='c1711783a9543aa6')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    frame_copy = np.empty((1080, 1920, 3), dtype=np.uint8)
    frame_inf = np.empty((224, 224, 3), dtype=np.uint8)


    #cv2.namedWindow("Live demo", cv2.WINDOW_AUTOSIZE) 
    cnt=0
    start = time.time()

    

    while not stop_event.is_set():

        # print("process processing")
        # get frame from buffer and do processing
        with lock:
            frame_copy[:,:,:] = np.copy(frame_buffer)



        frame_inf = cv2.resize(frame_copy, (224,224))
        frame_inf= cv2.cvtColor(frame_inf, cv2.COLOR_RGB2BGR)
        frame_copy= cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)


        outputs = rknn.inference(inputs=[frame_inf]) 
        K = 3
        frame_inf = transform_frame(outputs, frame_inf, K, conf=0.7)
        frame_copy = plot_keypoints(outputs, frame_copy, K, conf=0.7)


        end = time.time()
        cnt = cnt+1
        fps = cnt / (end-start)
        fps_text = f"FPS: {fps}"
        # display output frame on the terminal
        #sys.stdout.write('\033[3A')
        #frame_inf = cv2.resize(frame_inf,(500,500))
        # print("pp2")

        # add FPS info on the video stream
        # cv2.putText(frame_copy, fps_text, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_copy, fps_text, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_inf, fps_text, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #cv2.imshow('Live demo', frame_copy)
        cv2.imshow('Livedemo', frame_inf)


        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set() 
            break
        

       

if __name__ == "__main__":

    # realpath = os.path.abspath(__file__)
    # _sep = os.path.sep
    # realpath = realpath.split(_sep)
    # sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

    # from py_utils.pose_result_interpretor import transform_frame
    # from py_utils.coco_utils import COCO_test_helper

    # # setup phase
    # model_path = "/home/puffin/Desktop/rknn_model_zoo/examples/yolov8/model/yolov8s-pose.rknn"

    # rknn = RKNN(verbose=False)

    # print('--> Load model')
    # ret = rknn.load_rknn(path=model_path)
    # if ret != 0:
    #     print("Load rknn model failed")
    #     exit(ret)
    # print('done')

    # print('--> Init runtime environment')
    # ret = rknn.init_runtime(target='rk3588', device_id='c1711783a9543aa6')
    # if ret != 0:
    #     print('Init runtime environment failed!')
    #     exit(ret)
    # print('done')


    memmap_filename = "my_memmap_file.dat"
    memmap_path = os.path.join('/dev/shm', memmap_filename)
    # frame_shape = (480, 640, 3)  # resize the frame
    frame_shape = (1080, 1920, 3)  # resize the frame
    frame_dtype = 'uint8'
    stop_event = Event()
    lock = Lock()


    #cam set up
    # fpsf = cap.get(cv2.CAP_PROP_FPS)


    #frame_buffer = np.memmap('/dev/shm/readA', dtype = np.uint8, mode='w+', shape=frame_shape)
    frame_buffer = np.memmap('/home/puffin/Desktop/rknn_model_zoo/examples/yolov8/python/readA.dat', dtype = np.uint8, mode='w+', shape=frame_shape)

    p1 = Process(target=capture_frames, args=(frame_buffer, stop_event, lock))
    p2 = Process(target=process_frames, args=(frame_buffer, stop_event, lock))
    p1.start()
    p2.start()

    p1.join()
    p2.join()

