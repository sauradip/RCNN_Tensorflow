# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from libs.configs import cfgs

if cfgs.DATASET_NAME == 'ship':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'ship': 1
    }
elif cfgs.DATASET_NAME == 'FDDB':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'face': 1
    }
elif cfgs.DATASET_NAME == 'ICDAR2015':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'text': 1
    }

elif cfgs.DATASET_NAME == 'METTER':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'cs': 1
        # '0' : 2,
        # '1' : 3,
        # '2' : 4,
        # '3' : 5,
        # '4' : 6, 
        # '5' : 7,
        # '6' : 8,
        # '7' : 9,
        # '8' : 10,
        # '9' : 8,
        # # '9' : 11,
        # '01' : 11,
        # '12' : 12,
        # '23' : 13,
        # '34' : 14,
        # '45' : 15,
        # '56' : 16,
        # '67' : 17,
        # '78' : 18,
        # '89' : 19,
        # '90' : 20
    }

elif cfgs.DATASET_NAME == 'NUMBER':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        # 'cs': 1,
        '0' : 1,
        '1' : 2,
        '2' : 3,
        '3' : 4,
        '4' : 5, 
        '5' : 6,
        '6' : 7,
        '7' : 8,
        '8' : 9,
        '9' : 7,
        # '9' : 11,
        '01' : 10,
        '12' : 11,
        '23' : 12,
        '34' : 13,
        '45' : 14,
        '56' : 15,
        '67' : 16,
        '78' : 17,
        '89' : 18,
        '90' : 19
    }
elif cfgs.DATASET_NAME.startswith('DOTA'):
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'roundabout': 1,
        'tennis-court': 2,
        'swimming-pool': 3,
        'storage-tank': 4,
        'soccer-ball-field': 5,
        'small-vehicle': 6,
        'ship': 7,
        'plane': 8,
        'large-vehicle': 9,
        'helicopter': 10,
        'harbor': 11,
        'ground-track-field': 12,
        'bridge': 13,
        'basketball-court': 14,
        'baseball-diamond': 15
    }
elif cfgs.DATASET_NAME == 'pascal':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
    }
else:
    assert 'please set label dict!'


def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict

LABEl_NAME_MAP = get_label_name_map()
