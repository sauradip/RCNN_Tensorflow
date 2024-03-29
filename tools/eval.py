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
import json

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network
from help_utils.tools import *
from libs.box_utils import draw_box_in_img
from help_utils import tools
from libs.box_utils import coordinate_convert
from libs.label_name_dict.label_dict import LABEl_NAME_MAP

def inference(det_net, data_dir, anno_dir):

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

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = det_net.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        imgs = os.listdir(data_dir)
        for i, a_img_name in enumerate(imgs):

            # f = open('./res_icdar_r/res_{}.txt'.format(a_img_name.split('.jpg')[0]), 'w')
            raw_img = cv2.imread(os.path.join(data_dir,
                                              a_img_name))
            # raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

            start = time.time()
            resized_img, det_boxes_h_, det_scores_h_, det_category_h_, \
            det_boxes_r_, det_scores_r_, det_category_r_ = \
                sess.run(
                    [img_batch, det_boxes_h, det_scores_h, det_category_h,
                     det_boxes_r, det_scores_r, det_category_r],
                    feed_dict={img_plac: raw_img}
                )
            end = time.time()
            
            # res_r = coordinate_convert.forward_convert(det_boxes_r_, False)
            # res_r = np.array(res_r, np.int32)
            # for r in res_r:
            #     f.write('{},{},{},{},{},{},{},{}\n'.format(r[0], r[1], r[2], r[3],
            #                                                r[4], r[5], r[6], r[7]))
            # f.close()
            
            det_detections_h = draw_box_in_img.draw_box_cv(np.squeeze(resized_img, 0),
                                                           boxes=det_boxes_h_,
                                                           labels=det_category_h_,
                                                           scores=det_scores_h_)
            det_detections_r = draw_box_in_img.draw_rotate_box_cv(np.squeeze(resized_img, 0),
                                                                  boxes=det_boxes_r_,
                                                                  labels=det_category_r_,
                                                                  scores=det_scores_r_)
            eval_result(a_img_name, det_boxes_h_, det_category_h_, anno_dir, rotate = False)
            # eval_result(a_img_name, det_boxes_r_, det_category_r_, anno_dir, rotate = True)
            #  
            # print(det_boxes_r_, det_category_r_ )
            view_bar('{} cost {}s'.format(a_img_name, (end - start)), i + 1, len(imgs))
    print(NUMBER_BOX)
    print(SAME_LABELS)
    print('Simple Accuracy Number Box {} %'.format(round(np.mean(np.array(NUMBER_BOX)) * 100, 3)))
    print('Simple Accuracy Same Box {} %'.format(round(np.mean(np.array(SAME_LABELS)) * 100, 3)))

def eval_result(name, box, category, annodir, rotate = False):

    real_label = []
    jsonpath = os.path.join(annodir, name[:-4] + '.json')
    with open(jsonpath, 'r') as f:
        data = json.load(f)
        point_data = data["shapes"]
        for item in point_data:
            real_label.append(item["label"])
    target = len(real_label)

    if not rotate:
        boxes = box.astype(np.int64)
        labels = category.astype(np.int32)
        labels = [LABEl_NAME_MAP[i] for i in labels]
        for label in labels:
            if label in real_label:
                real_label.remove(label)
        
        accuracy_number_box = len(labels) / target
        accuracy_same =  1  - len(real_label) / target
        NUMBER_BOX.append(accuracy_number_box)
        SAME_LABELS.append(accuracy_same)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a R2CNN network')
    parser.add_argument('--img_dir', dest='data_dir',
                        help='data path', type=str)
    parser.add_argument('--anno_dir', 
                        help='data path', type=str)                
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu index',
                        default='0', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    NUMBER_BOX = []
    SAME_LABELS = []
    args = parse_args()
    print('Called with args:')
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    det_net = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                   is_training=False)

    inference(det_net, data_dir=args.data_dir, anno_dir = args.anno_dir )

















