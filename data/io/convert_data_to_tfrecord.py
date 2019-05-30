# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys
sys.path.append('../../')
import xml.etree.cElementTree as ET
import numpy as np
import tensorflow as tf
import glob
import cv2
from libs.label_name_dict.label_dict import *
from help_utils.tools import *
from libs.configs import cfgs
import json

tf.app.flags.DEFINE_string('data_dir', '/root/userfolder/yx/', 'Voc dir')
tf.app.flags.DEFINE_string('json_dir', 'icdar2015_xml', 'xml dir')
tf.app.flags.DEFINE_string('image_dir', 'icdar2015_img', 'image dir')
tf.app.flags.DEFINE_string('save_name', 'train', 'save name')
tf.app.flags.DEFINE_string('save_dir', '../tfrecord/', 'save name')
tf.app.flags.DEFINE_string('img_format', '.jpg', 'format of image')
tf.app.flags.DEFINE_string('dataset', 'ICDAR2015', 'dataset')
FLAGS = tf.app.flags.FLAGS

label_list = []

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def read_json_gtbox_and_label(json_path):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 9],
           and has [x1, y1, x2, y2, x3, y3, x4, y4, label] in a per row
    """
    boxlist = []
    with open(json_path) as f:
        data = json.load(f)

        img_height = data["imageHeight"]
        img_width = data["imageWidth"]
        point_data = data["shapes"]
        for item in point_data:
            if item['label'] in NAME_LABEL_MAP :
                label = NAME_LABEL_MAP[item['label']]
                if label not in label_list:
                    label_list.append(label)

                point = list(np.array(item['points']).flatten())
                point.append(label)
                # print(point)        
                boxlist.append(point)
                # print(point)
        # print('height, weight',img_height, img_width)
        # print('box list',np.array(boxlist,dtype=np.int32))

    return img_height, img_width, np.array(boxlist,dtype=np.int32)

def convert_pascal_to_tfrecord():
    json_path = FLAGS.data_dir + FLAGS.json_dir
    image_path = FLAGS.data_dir + FLAGS.image_dir
    save_path = FLAGS.save_dir + cfgs.VERSION + '_' +  FLAGS.dataset + '_' + FLAGS.save_name + '.tfrecord'
    print(save_path)
    mkdir(FLAGS.save_dir)
    # writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    # writer = tf.python_io.TFRecordWriter(path=save_path, options=writer_options)
    writer = tf.python_io.TFRecordWriter(path=save_path)
    # print(json_path)
    for count, json in enumerate(glob.glob(json_path + '/*.json')):
        print(json)
        # to avoid path error in different development platform
        json = json.replace('\\', '/')

        img_name = json.split('/')[-1].split('.')[0] + FLAGS.img_format
        img_path = image_path + '/' + img_name

        if not os.path.exists(img_path):
            print('{} is not exist!'.format(img_path))
            continue
        # read_json_gtbox_and_label(json)
        img_height, img_width, gtbox_label = read_json_gtbox_and_label(json)
        print('gtbox shape' ,gtbox_label.shape)
        print('gtbox ' ,gtbox_label)
        if gtbox_label.shape[0] == 0:
            continue
        # img = np.array(Image.open(img_path))
        img = cv2.imread(img_path)

        feature = tf.train.Features(feature={
            # do not need encode() in linux
            'img_name': _bytes_feature(img_name.encode()),
            # 'img_name': _bytes_feature(img_name),
            'img_height': _int64_feature(img_height),
            'img_width': _int64_feature(img_width),
            'img': _bytes_feature(img.tostring()),
            'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
            'num_objects': _int64_feature(gtbox_label.shape[0])
        })

        example = tf.train.Example(features=feature)

        writer.write(example.SerializeToString())

        view_bar('Conversion progress', count + 1, len(glob.glob(json_path + '/*.json')))
    
    # print(label_list)
    
    print('\nConversion is complete!')


if __name__ == '__main__':
    # xml_path = 'test.xml'
    # read_xml_gtbox_and_label(xml_path)

    convert_pascal_to_tfrecord()
