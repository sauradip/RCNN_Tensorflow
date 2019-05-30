import os
import glob
import json
import cv2 as cv
import numpy as np

img_dir = '/home/khach/hanh/R2CNN_Faster-RCNN_Tensorflow/metter_data/images_resize'
json_dir = '/home/khach/hanh/R2CNN_Faster-RCNN_Tensorflow/metter_data/json_resize'

save_image_dir = '/home/khach/hanh/R2CNN_Faster-RCNN_Tensorflow/metter_data/images_resize_2'
save_json_dir = '/home/khach/hanh/R2CNN_Faster-RCNN_Tensorflow/metter_data/json_resize_2'
def resize_with_pad(image, jsonpath, basename):

    def get_padding_size(image):

        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv.copyMakeBorder(image, top , bottom, left, right, cv.BORDER_CONSTANT, value=BLACK)

    resized_image = cv.resize(constant, (1000, 1000), cv.INTER_NEAREST)
    print(constant.shape[0])
    scale = 1000.0 / float(constant.shape[0])
    # print(scale)
    print(top , bottom, left, right)
    newjson = os.path.join(save_json_dir, '{}.json'.format(basename))
    # newjsonfile = open(newjson, "rw")

    jsonfile = open(jsonpath, "r")
    data = json.load(jsonfile)

    shapes = data["shapes"]
    for item in shapes:
        # print(item['points'])
        tmp = []
        for point in item['points']:
            newpoint = [int(point[0] * scale) , int(point[1] * scale) ]
            tmp.append(newpoint)
        # print(tmp)
        # exit()
        item['points'] = tmp
    with open(newjson, "w") as f:
        json.dump(data, f)

    return constant 

if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)
if not os.path.exists(save_json_dir):
    os.makedirs(save_json_dir)

imglist = sorted(glob.glob('{}/*'.format(img_dir)))
for img in imglist:
    basename = os.path.splitext(os.path.basename(img))[0]
    jsonpath = os.path.join(json_dir, '{}.json'.format(basename))
    image  = cv.imread(img)
    imgpad = resize_with_pad(image, jsonpath, basename)
    
    
    cv.imwrite('{}/{}.jpg'.format(save_image_dir,basename), imgpad)
    # exit()