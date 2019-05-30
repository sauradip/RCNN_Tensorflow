from __future__ import division
import os
import glob
import sys
sys.path.append('../')
from libs.label_name_dict.label_dict import NAME_LABEL_MAP
import json
import numpy as np 
import shutil
# DIR_PRE  = '/media/Hard_drive/R2CNN/output/evaluate_h_result_pickle/10class_20190425'
DIR_IMAGES = '/home/ekgis/hanh/R2CNN/metter/number/test/images/'
#get result from predict
def caculate(DIR_PRE, IMGLIST):
    predict_dict = {}
    predict = []
    for key in NAME_LABEL_MAP:
        if key == 'back_ground':
            continue
        path = os.path.join(DIR_PRE,'det_' + key + '.txt')
        with open(path, mode = 'r') as f:
            datas = f.readlines()
            for item in datas:
                item = item.split('\n')[0] + ' ' + key
                predict.append(item)

    for item in predict:
        element = item.split(' ')
        name = element[0]
        number = element[-1]
        if element[0] not in predict_dict:
            predict_dict.update({'{}'.format(element[0]):[]})
        
        predict_dict[name].append(number)
    # print(predict_dict)
    #get result from json file
    # imglist = sorted(glob.glob('{}/*'.format(DIR_IMAGES)))
    imglist = IMGLIST
    objects = {}
    for img in imglist:
        basename = img
        imgpath = os.path.join(DIR_IMAGES, basename +'.jpg' )
        jsonpath = imgpath.replace('/images','/json').replace('.jpg','.json')

        if basename not in objects:
            objects.update({'{}'.format(basename):[]})

        with open(jsonpath) as f:
            data = json.load(f)
            point_data = data["shapes"]
            # obj_struct = {}
            for item in point_data:
                if item['label'] in NAME_LABEL_MAP :
                    objects[basename].append(str(item['label']))
    # print(objects)
    # print(predict_dict)
    # exit()
    accuracy = []
    box = []
    for item in predict_dict:
        
        predict_classes = predict_dict[item]
        true_classes  = objects[item]
        hold_true = len(true_classes)
        # print('hold true', item, objects[item] , hold_true)
        # if hold_true == 0:
        #     imgpath = os.path.join(DIR_IMAGES, item + '.jpg' )
        #     jsonpath = imgpath.replace('/images/', '/json/').replace('.jpg','.json')
        #     shutil.copy2(imgpath, '/home/ekgis/hanh/R2CNN/metter/20190426/test/' )
        #     shutil.copy2(jsonpath, '/home/ekgis/hanh/R2CNN/metter/20190426/test/' )
        count = 0
        for i, clap in enumerate(predict_classes):
            for j , clat in enumerate(true_classes):
                if predict_classes[i] == true_classes[j]:
                    count +=1
                    del true_classes[j]
                
                continue
        if hold_true == 0:
            box.append(0)
            accuracy.append(0)
        else:
            box.append(len(predict_classes)/hold_true)
            accuracy.append(count/hold_true)
        print('hold true', item, objects[item] , hold_true, len(predict_classes) )
    # print(accuracy)
    print('Simple Accuracy {} %'.format(round(np.mean(np.array(box)) * 100, 3)))
    print('Simple Accuracy {} %'.format(round(np.mean(np.array(accuracy)) * 100, 3)))
