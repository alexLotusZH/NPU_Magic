import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from ultralytics.utils.ops import scale_coords
import sys
import os

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (147, 20, 255),
    (0, 2): (255, 255, 0),
    (1, 3): (147, 20, 255),
    (2, 4): (255, 255, 0),
    (0, 5): (147, 20, 255),
    (0, 6): (255, 255, 0),
    (5, 7): (147, 20, 255),
    (7, 9): (147, 20, 255),
    (6, 8): (255, 255, 0),
    (8, 10): (255, 255, 0),
    (5, 6): (0, 255, 255),
    (5, 11): (147, 20, 255),
    (6, 12): (255, 255, 0),
    (11, 12): (0, 255, 255),
    (11, 13): (147, 20, 255),
    (13, 15): (147, 20, 255),
    (12, 14): (255, 255, 0),
    (14, 16): (255, 255, 0)
}


class Pose_output():
    def __init__(self, outputs, pic_path, K = 3, size = (224,224)):
        array = np.array(outputs)
        array = array.squeeze()
        self.tensor = array.T
        self.img_path = pic_path
        img = cv2.imread(pic_path)
        img = cv2.resize(img, size)
        self.height, self.width, _= img.shape
        self.image = img
        self.k = K

    def draw_bbox_on_image(self, x, y, w, h):
        # Denormalize the coordinates
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        # Draw the bounding box
        # Calculate the (x1, y1) and (x2, y2) points for the rectangle
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2

        # Draw the bounding box
        cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def plot_keypoints_on_image(self, keypoints, t):
        for edge, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            point1_index, point2_index = edge
            x1, y1, _ = keypoints[point1_index]
            x2, y2, _ = keypoints[point2_index]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw the line on the image
            cv2.line(self.image, (x1, y1), (x2, y2), color, 2)
            cv2.circle(self.image, (x1, y1), 4, color, -1) 

    def print_top_keypoints(self):
        sorted_indices = np.argsort(self.tensor[:,4])[::-1]
        # print(self.tensor[:,4])
        top_K_by_confidence = self.tensor[sorted_indices[0:self.k]]
        print("top_K_by_confidence", top_K_by_confidence[0])
        
        for bbox in top_K_by_confidence:
            # Select the first 51 elements and reshape it into 17x3
            keypoints = bbox[5:].reshape((17, 3))
            xywh = bbox[:4]
            self.draw_bbox_on_image(xywh[0], xywh[1], xywh[2], xywh[3])
            self.plot_keypoints_on_image(keypoints, 0.7)

        base_name, ext = os.path.splitext(self.img_path)
        save_name = f"{base_name}_boxed{ext}"

        # resize to its original size
        # self.image = cv2.resize(self.image, (self.width, self.height))
        cv2.imwrite(save_name, self.image)

def draw_bbox_on_image(img, x, y, w, h, scale_x=1, scale_y=1):
    # Denormalize the coordinates
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    x, y, w, h = int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)


    # Draw the bounding box
    # Calculate the (x1, y1) and (x2, y2) points for the rectangle
    x1, y1 = x - w // 2, y - h // 2
    x2, y2 = x + w // 2, y + h // 2


    # Draw the bounding box
    return cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

def plot_keypoints_on_image(img, keypoints, conf, scale_x=1, scale_y=1):
    for edge, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        point1_index, point2_index = edge
        x1, y1, cf1 = keypoints[point1_index]
        x2, y2, cf2 = keypoints[point2_index]

        x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
        
        # Draw the line on the image
        if (cf1 > conf and cf2 > conf):
            img = cv2.line(img, (x1, y1), (x2, y2), color, 2)
            img = cv2.circle(img, (x1, y1), 4, color, -1) 

    return img


#Intersection of Union helper function
def calculate_iou(box1, box2):
#    """ip addr       
    

#         box1 (list): [x, y, w, h]   
#         box2 (list): [x, y, w, h]
        
#     return:
#         value of IOU (float)
#     """

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
 
    rect1 = (x1, y1, x1 + w1, y1 + h1)
    rect2 = (x2, y2, x2 + w2, y2 + h2)
    

    intersection = (max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])) *
                    max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1])))
    

    union = w1 * h1 + w2 * h2 - intersection
    
 
    iou = intersection / union if union > 0 else 0
    
    return iou

#MAr 14 Checkpoint
def transform_frame(output, img, K, conf=0.7, size=(224,224)):
    array = np.array(output)
    array = array.squeeze()
    tensor = array.T
    sorted_indices = np.argsort(tensor[:,4])[::-1]
    top_K_by_confidence = tensor[sorted_indices[0:K]]
    
    i = 0
    xywh_prev = np.array([0.0,0.0,0.0,0.0])
    box_printed = 0
    while i < K and K < len(sorted_indices):

        bbox = tensor[sorted_indices[:]][i]
        #print('working on', i,' ', K)
        keypoints = bbox[5:].reshape((17, 3))

        xywh = bbox[:4]
        for edge, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            point1_index, point2_index = edge
            _, _, cf1 = keypoints[point1_index]
            _, _, cf2 = keypoints[point2_index]

        iou = calculate_iou(xywh,xywh_prev)
        xywh_prev = xywh
        is_overlap =  iou> 0.8
        if is_overlap:
            K+=1
            i+=1
            print("skipped overlapping i:", i, ' ,K:', K, " ,iou: ",iou )
            continue
        if (bbox[4] > conf):
            img = draw_bbox_on_image(img, xywh[0], xywh[1], xywh[2], xywh[3])
            img = plot_keypoints_on_image(img, keypoints, conf)
        box_printed +=1
        i+=1
        print("box printed: ", box_printed)


    # for bbox in top_K_by_confidence:
    #     # Select the first 51 elements and reshape it into 17x3
    #     keypoints = bbox[5:].reshape((17, 3))
    #     xywh = bbox[:4]

    #     img = draw_bbox_on_image(img, xywh[0], xywh[1], xywh[2], xywh[3])
    #     img = plot_keypoints_on_image(img, keypoints, conf)

    return img

def plot_keypoints(output, img, K=3, conf=0.7, size=(224,224)):
    array = np.array(output)
    array = array.squeeze()
    tensor = array.T

    sorted_indices = np.argsort(tensor[:,4])[::-1]
    top_K_by_confidence = tensor[sorted_indices[0:K]]
    width,height,_ = img.shape
    scale_x = float(height) / size[0]
    scale_y = float(width) / size[1]

    
    for bbox in top_K_by_confidence:
        # Select the first 51 elements and reshape it into 17x3
        keypoints = bbox[5:].reshape((17, 3))
        xywh = bbox[:4]
        img = draw_bbox_on_image(img, xywh[0], xywh[1], xywh[2], xywh[3], scale_x, scale_y)
        img = plot_keypoints_on_image(img, keypoints, conf, scale_x, scale_y)

    return img