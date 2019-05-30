import os
import numpy as np
import glob
from math import sqrt
from libs.configs import cfgs
import cv2
from Classification.utils.add_padding import resize_with_pad
from classification_model import classifer

NAME_LABEL_MAP = {
        '0' : 'back_ground',
        '1' : 'cs',
        '2' : '0' ,
        '3' : '1' ,
        '4' : '2' ,
        '5' : '3' ,
        '6' : '4' , 
        '7' : '5' ,
        '8' : '6' ,
        '9' : '7' ,
        '10': '8',
}
boxes = np.array([[ 49.867004, 286.30957,  128.80733,  486.88855 ],
 [ 59.83148 , 445.1297,   105.60469,  471.82098 ],
 [ 60.81122 , 413.28824 , 108.73154 , 442.6293  ],
 [ 65.6604 ,  386.75204 , 107.26308 , 412.9213  ],
 [ 69.941025, 356.64404 , 111.88068 , 380.85327 ]])

labels = np.array([ 1.,  2.,  3.,  6., 10.])

def save_preprocess(img, boxes, labels, name, des, crop):
# def save_preprocess(boxes, labels):

    centers = []
    lab = []
    imgs = []

    horizonal = True
    coor_metter = None
    label_metter = None
    
    # print(name)
    boxs = boxes.astype(np.int64)
    labels = labels.astype(np.int32)
    img = img + np.array(cfgs.PIXEL_MEAN)
    img = np.array(img, np.float32)
    img = np.array(img*255/np.max(img), np.uint8)
    if 1 not in labels:
        return
    
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        new_rectangle = np.array([[xmin, ymin],[xmax,ymin],[xmax, ymax],[xmin, ymax]])

        center = np.array([(xmax + xmin)/2, (ymax + ymin)/2])
        label = labels[i]
        if label == 0 :
            continue
        
        if label == 1 : 
            coor_metter = np.array([xmin, ymin, xmax, ymax])
            label_metter = NAME_LABEL_MAP[str(label)]
            continue
        
        centers.append(center)
        lab.append(NAME_LABEL_MAP[str(label)])

        if crop :
            tmp = []
            new_rectangle = new_rectangle.reshape(new_rectangle.shape[0], 1 ,new_rectangle.shape[1])
            rect = cv2.minAreaRect(new_rectangle)

            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # print("bounding box: {}".format(box))
            # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

            # img_crop will the cropped rectangle, img_rot is the rotated image
            img_crop, img_rot = crop_rect(img, rect)
            
            img = resize_with_pad(img_crop, 48, 48 )
            imgs.append(img.astype('float32') / 255)
            # filter_lab = classifer(tmp)
            # img_crop_path = os.path.join(des, 'crop', str(label - 2))
            
            # if not os.path.exists(img_crop_path):
            #     os.makedirs(img_crop_path)

            # cv2.imwrite("{}/{}.jpg".format(img_crop_path, name + '{}'.format(i)), img_crop)

    filter_lab = [str(element) for element in classifer(imgs)]

    centers = np.array(centers)
    coor_metter = np.array(coor_metter)
    print('original',centers)
    max_vertical = np.amax(centers[:,1])
    max_horizonal = np.amax(centers[:,0])
    min_vertical = np.amin(centers[:,1])
    min_horizonal = np.amin(centers[:,0])
    print('denta y', max_vertical - min_vertical)
    print('denta x', max_horizonal - min_horizonal)
    if max_vertical - min_vertical > max_horizonal - min_horizonal :
        horizonal = False

    
    result_normal = caculate_result(centers, coor_metter, lab, horizonal)  
    result_filter = caculate_result(centers, coor_metter, filter_lab, horizonal)

    return result_normal, result_filter

def caculate_result(centers, coor_metter, labels ,horizonal ):
    # coor_metter xmin, ymin, xmax, ymax
    distance = []
    newlabels = []
    final_labels=[]
    if horizonal:
        print('horizonal')
        centers = centers[:,0].tolist()
        print(centers)
        print(labels)
        xy = zip(centers, labels)
        xy = sorted(xy, key= lambda x : x[0])
        centers = [i for i, j in xy ]
        labels = [j for i, j in xy]

        centers.append(coor_metter[2])
        centers.insert(0, coor_metter[0])

    else:
        print('vertical')
        centers = centers[:,1].tolist()
        print(centers)
        print(labels)
        xy = zip(centers, labels)
        xy = sorted(xy, key= lambda x : x[0])
        centers = [i for i, j in xy ]
        labels = [j for i, j in xy]

        centers.append(coor_metter[3])
        centers.insert(0, coor_metter[1])
    
    # print('final centers', centers)
    # print('label', labels)

    for i in range(len(centers)-1):
        distance.append(centers[i+1] - centers[i])
    print('distance', distance)

    if len(distance) == 2:
        ratio = distance[0]/distance[1]
        numberx_first = int(5 * ratio / (ratio + 1 ))
        numberx_behind = 5 - numberx_first
        for i in range(numberx_first):
            newlabels.append('x')
        newlabels.append(labels[0])
        for j in range(numberx_behind):
            newlabels.append('x')

        return newlabels

    minvalue = min(distance[1:-1])
    newdis = distance[1:-1]
    newlabels.append(labels[0])
    for i, item in enumerate (newdis):
        if item / minvalue > 2:
            for j in range( int(item/minvalue)):
                newlabels.append('x')
            newlabels.append(labels[i+1])
        else:
            newlabels.append(labels[i+1])

    off_set_first = distance[0] / minvalue
    off_set_behind = distance[-1] / minvalue
    if int(off_set_first) >= 1:
        for i in range(int(off_set_first)):
            final_labels.append('x')
    final_labels = final_labels + newlabels
    if int(off_set_behind) >= 2:
        for i in range(int(off_set_behind)-1):
            final_labels.append('x')
    print(final_labels)
    # if len(newlabels) < 6 :
    #     for k in range(6-len(newlabels)):
    #         newlabels.append('x')
    
    return newlabels
    

def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot

# if __name__ == "__main__":
#     save_preprocess(boxes, labels)