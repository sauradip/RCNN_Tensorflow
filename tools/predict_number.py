# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys
sys.path.append("../")
import tensorflow as tf
import time
import cv2
import numpy as np
import argparse

from libs.label_name_dict.label_dict import LABEl_NAME_MAP
from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network
from help_utils.tools import *
from libs.box_utils import draw_box_in_img
from help_utils import tools
from libs.box_utils import coordinate_convert
# from utils.rotate import process
import glob

def inference(det_net, data_path):

    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
    img_batch = tf.cast(img_plac, tf.float32)
    img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN)
    det_boxes_h, det_scores_h, det_category_h, \
    det_boxes_r, det_scores_r, det_category_r = det_net.build_whole_detection_network(input_img_batch=img_batch,
                                                                                      gtboxes_h_batch=None,
                                                                                      gtboxes_r_batch=None)
    fininsh_load = time.time()
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = det_net.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False

    tmp = []

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        # imgs = os.listdir(data_dir)
        # for i, a_img_name in enumerate(imgs):

        #     raw_img = cv2.imread(os.path.join(data_dir,
        #                                       a_img_name))
        raw_img = cv2.imread(data_path)
        basename = os.path.splitext(os.path.basename(data_path))[0]

        start = time.time()
        resized_img, det_boxes_h_, det_scores_h_, det_category_h_, \
        det_boxes_r_, det_scores_r_, det_category_r_ = \
            sess.run(
                [img_batch, det_boxes_h, det_scores_h, det_category_h,
                    det_boxes_r, det_scores_r, det_category_r],
                feed_dict={img_plac: raw_img}
            )
        end = time.time()

        boxes = det_boxes_h_
        category = det_category_h_
        scores = det_scores_r_

        process_data(boxes, category, scores,  np.squeeze(resized_img, 0), basename )


def process_data(boxs, categories, scores, img, basename):
    
    img = img + np.array(cfgs.PIXEL_MEAN)
    img = np.array(img, np.float32)
    img = np.array(img*255/np.max(img), np.uint8)
    
    categories = [LABEl_NAME_MAP[i] for i in categories]
    # caculate the average width
    x_min_all = np.amin(boxs[:, 0])
    x_max_all = np.amax(boxs[:, 0])
    average_width = (x_max_all - x_min_all)/6 - 15
    print(average_width)

    #caculate overlab base on center
    center_x = [(box[0] +box[2])/2  for box in boxs]
    center_x, categories, boxs, scores = zip(*sorted(zip(center_x, categories, boxs, scores), key=lambda center_x : center_x[0]))
    print(categories)
    print(scores)
    labels_new, boxs_new, scores_new = check_overlap(list(center_x), list(categories), list(boxs), 
                                                        list(scores), average_width )
    path = cfgs.TEST_PATH + basename
    if not os.path.exists(path):
        os.makedirs(path)
    #make result
    for idx, item in enumerate(labels_new):
        box = boxs_new[idx]
        label = item
        xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        img_ = img[ymin : ymax, xmin : xmax]
        cv2.imwrite('{}/{}_{}_{}.jpg'.format(path, basename, item, idx), img_)

def check_overlap(center, labels, boxs, scores, average_width):
    center_new = []
    labels_new = []
    boxs_new = []
    scores_new = []
    stop = False
    idx = 0
    print('len center', len(center))
    while not stop:
        print(idx , len(center))
        if idx + 1 == len(center):
            labels_new.append(labels[idx])
            boxs_new.append(boxs[idx])
            scores_new.append(scores[idx])
            stop = True
            break
            
        if idx + 1 > len(center):
            print('stop')
            stop = True
            break

        distance = center[idx + 1] - center[idx]
        if distance < average_width:
            if scores[idx+1] > scores[idx]:
                labels_new.append(labels[idx + 1])
                boxs_new.append(boxs[idx + 1])
                scores_new.append(scores[idx + 1])
                
                del center[idx]
                del labels[idx]
                del boxs[idx]
                del scores[idx]
                print('tmp',len(center))

            else:
                labels_new.append(labels[idx])
                boxs_new.append(boxs[idx])
                scores_new.append(scores[idx])

                del center[idx + 1]
                del labels[idx + 1]
                del boxs[idx + 1]
                del scores[idx + 1]
                print('tmp',len(center))
        else:
            labels_new.append(labels[idx])
            boxs_new.append(boxs[idx])
            scores_new.append(scores[idx])


        idx +=1    
    
    return labels_new, boxs_new, scores_new

        
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a R2CNN network')
    parser.add_argument('--data_path',  type=str)
   
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu index',
                        default='0', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    det_net = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                   is_training=False)

    inference(det_net, args.data_path)

















