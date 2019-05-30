import os
import numpy as np
import glob
import cv2 as cv
import json
import sys
sys.path.append('../')
import base64

DIR = 'D:\\Project\\Research\\Detection\\R2CNN\\metter\\20190426'
# DIR = 'D:\\Project\\Research\\Detection\\R2CNN\\metter\\test_cs'
DES = 'D:\\Project\\Research\\Detection\\R2CNN\\metter\\sample'
# DES = 'D:\\Project\\Research\\Detection\\R2CNN\\metter\\horizolcs\\test'
CLASS = 'D:\\Project\\Research\\Detection\\R2CNN\\metter\\sample'
# LABEL_DICT = ['0' , '1' ,'2' , '3' , '4' , '5' ,'6' , '7' , '8' , '9' ]
# LABEL_DICT = ['cs']

def load():
    jsonfiles = sorted(glob.glob('{}/json/*.json'.format(DIR)))

    for json in jsonfiles:
        jsonname = os.path.splitext(os.path.basename(json))[0]
        imgname = jsonname + '.jpg'
        imgpath = os.path.join(DIR + '/images/' ,imgname)
        if os.path.exists(imgpath):
            #get data
            print(json)
            print(imgpath)
            img, box, labels = getdata(imgpath, json, jsonname)
            #crop box
            if img is not None:
                #get data in the cs image:
                img, box, labels = crop_image(img, box, labels)
                
                #make json file for only metter:
                # make_json(DES, jsonname, img, box, labels)
                
                #make classification data:
                make_class(DES, jsonname, img, box, labels)
def make_class(DES, basename, img, boxes, labels):
    xuoi_txt = 'D:\\Project\\Research\\Detection\\R2CNN\\utils\\xuoi.txt'
    nguoc_txt = 'D:\\Project\\Research\\Detection\\R2CNN\\utils\\nguoc.txt'
    xuoi = []
    nguoc = []
    xuoi_flag = True
    with open(xuoi_txt, 'r') as f :
        xuoi_file = f.readlines()
        # print(xuoi_file)
        for x_file in xuoi_file:
            xuoi.append(x_file.strip())    
    with open(nguoc_txt, 'r') as f1 :
        nguoc_file = f1.readlines()
        # print(xuoi_file)
        for n_file in nguoc_file:
            nguoc.append(n_file.strip())    

    if basename not in xuoi:
        xuoi_flag = False

    for idx , label in enumerate(labels):
        if label != 'cs':
            if xuoi_flag:
                img_dir = os.path.join(CLASS, label)
            else:
                img_dir = os.path.join(CLASS, label + '_rotate')
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            box = boxes[idx]
            minx, miny, maxx, maxy = int(np.amin(box[:,0])), int(np.amin(box[:,1])), int(np.amax(box[:,0])), int(np.amax(box[:,1]))
            img_ = img[miny : maxy, minx : maxx]
            # cv.imshow('a',img)
            # cv.waitKey(0)
            imgpath = '{}/{}_{}.jpg'.format(img_dir, basename, idx)
            print(imgpath)
            cv.imwrite(imgpath, img_)

def caculate_box( box, min_x , min_y, max_x, max_y):
        s_min_x , s_min_y, s_max_x, s_max_y = np.amin(box[:,0]) , np.amin(box[:,1]) ,np.amax(box[:,0]) , np.amax(box[:,1])
        
        tmp_x_min = s_min_x - min_x if s_min_x > min_x else 0
        tmp_y_min = s_min_y - min_y if s_min_y > min_y else 0
        tmp_x_max = s_max_x - min_x if max_x > s_max_x else max_x
        tmp_y_max = s_max_y - min_y if max_y > s_max_y else max_y
        
        return [[tmp_x_min, tmp_y_min],[tmp_x_min, tmp_y_max],[tmp_x_max, tmp_y_max],[tmp_x_max, tmp_y_min]]

def crop_image(img, box, labels) :
    result = []
    real_labels = []
    for idx, item in enumerate(box):
        if labels[idx] == 'cs':
            min_x , min_y, max_x, max_y = np.amin(box[idx][:,0]) , np.amin(box[idx][:,1]) , np.amax(box[idx][:,0]) , np.amax(box[idx][:,1])
            min_x = int(min_x)
            min_y = int(min_y)
            max_x = int(max_x)
            max_y = int(max_y)
            # new = np.array([[min_x, min_y],[min_x, max_y],[max_x, max_y], [max_x, min_y]])
            # cv.drawContours(img, [new.astype(int)], -1, (255,0,0), 2)
            # cv.imshow('a',img)
            # cv.waitKey(0)

    img = img[min_y : max_y, min_x : max_x]
    # print(img_.shape)
    # cv.imshow('abcccc',img)
    # cv.waitKey(0)
    for id_, item_ in enumerate(box):
        if labels[id_] != 'cs':
            box_ = caculate_box(box[id_], min_x , min_y, max_x, max_y)
            real_labels.append(labels[id_])
            # cv.drawContours(img, [np.array(box_).astype(int)], -1, (255,0,0), 2)
            result.append(np.array(box_))
    # cv.imshow('a',img)
    # cv.waitKey(0)

    return img, np.array(result), real_labels

def make_json(DES, basename, img, box, labels):

    imgpath = os.path.join(DES , 'images' )
    if not os.path.exists(imgpath):
        os.makedirs(imgpath)

    cv.imwrite('{}/{}.jpg'.format(imgpath, basename), img)

    img = cv.imread('{}/{}.jpg'.format(imgpath, basename))
    h, w, c = img.shape
    
    with open('{}/{}.jpg'.format(imgpath, basename), 'rb') as f:
        imageData = f.read()
        imageData = base64.b64encode(imageData).decode('utf-8')

    new_json = {}
    new_json["version"] = "3.11.2"
    new_json["flags"] = {}
    new_json["imageWidth"] = w
    new_json["imageHeight"] = h

    new_json["lineColor"] = [ 0, 255, 0, 128 ]
    new_json["imagePath"] = basename + '.jpg'
    new_json["fillColor"] = [255, 0, 0, 128]
    new_json["shapes"] = []
    new_json["imageData"] = imageData
    
    json_path = os.path.join('{}\\json'.format(DIR), basename + '.json')
    with open(json_path) as f:
        data = json.load(f)
        point_data = data["shapes"]
        for idx, point in enumerate(box):
            tmp = {}
            tmp["shape_type"] = 'polygon'
            tmp["line_color"] = None
            tmp["fill_color"] = None
            tmp["label"] = labels[idx]

            tmp["points"] = np.array(point).astype(int).tolist()
            new_json["shapes"].append(tmp)
    
    jsonpath = os.path.join(DES , 'json' )
    if not os.path.exists(jsonpath):
        os.makedirs(jsonpath)

    with open('{}/{}.json'.format(jsonpath, basename), 'w') as outfile:  
        json.dump(new_json, outfile) 
    


def getdata(imgpath, jsonpath, basename):
    
    img = cv.imread(imgpath)
    (heigth, width) = img.shape[:2]
    (cx, cy) = (width // 2, heigth // 2)

    labels = []
    new_labels = []
    new_box = []

    rotate_again = False
    final_image = None
    final_box = []
    final_labels = []

    with open(jsonpath) as f:
        data = json.load(f)
        # get rotated image
        point_data = data["shapes"]

        for idx, item in enumerate(point_data):
            label = item['label']
            labels.append(label)

        if 'cs' not in labels:
            return None, None, None

        for idx, item in enumerate(point_data):
            label = item['label']
            if label == 'cs':
                points = np.array(item['points'])
                rect = cv.minAreaRect(points)
                (csx, csy),(h,w), theta = cv.minAreaRect(points)
                rotated = rotate_bound(img, theta)

                (new_h, new_w) = rotated.shape[:2]
                (new_cx, new_cy) = (new_w // 2, new_h // 2)

        for idx, item in enumerate(point_data):
            label = item['label']
            points = np.array(item['points'])
            # cv.drawContours(img, [points], -1, (0,255,0), 2)

        #draw rotated image

        for idx, item in enumerate(point_data):
            label = item['label']
            points = np.array(item['points'])
            newpoint = rotate_box(points, cx, cy, heigth, width, theta )
            
            new_labels.append(label)
            new_box.append(np.array(newpoint))

            # cv.drawContours(rotated, [np.array(newpoint).astype(int)], -1, (0,255,0), 2)

        new_labels = np.array(new_labels)
        new_box = np.array(new_box)

        # check the direction of box
        
        for idx, item in enumerate(new_box):
            if new_labels[idx] == 'cs':
                distance_x = np.amax(new_box[idx][:,0]) - np.amin(new_box[idx][:,0])
                distance_y = np.amax(new_box[idx][:,1]) - np.amin(new_box[idx][:,1])
                if distance_x < distance_y:
                    rotate_again = True
        
        if rotate_again:
            new_img = rotate_bound(rotated, 90)

            for idx, item in enumerate(new_box):
                label = new_labels[idx]
                new_point = rotate_box(item, new_cx, new_cy, new_h, new_w, 90 )
                
                # cv.drawContours(new_img, [np.array(new_point).astype(int)], -1, (0,255,0), 2)

                final_box.append(np.array(new_point))
                final_labels.append(label)
            
            # cv.imshow('out',new_img)
            # cv.waitKey(0)
            return new_img, final_box , final_labels
        else:
            # cv.imshow('out', rotated)
            # cv.waitKey(0)
            return rotated, new_box, new_labels
            

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv.warpAffine(image, M, (nW, nH))

def rotate_box(bb, cx, cy, h, w, theta):

    new_bb = list(bb)
    for i,coord in enumerate(bb):
        # opencv calculates standard transformation matrix
        M = cv.getRotationMatrix2D((cx, cy), theta, 1.0)
        # Grab  the rotation components of the matrix)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        v = [coord[0],coord[1],1]
        # Perform the actual rotation and return the image
        calculated = np.dot(M,v)
        new_bb[i] = (calculated[0],calculated[1])
    
    return new_bb

if __name__ == "__main__":
    load()
    # des1 = 'D:\\Project\\Research\\Detection\\R2CNN\\metter\\20190426\\images'
    # des2 = 'D:\\Project\\Research\\Detection\\R2CNN\\metter\\cs\\test\\images'
    # with open('xuoi.txt', 'a') as f:
    #     imgnguoc = sorted(glob.glob('{}/*'.format(des1)))
    #     for imgn in imgnguoc:
    #         basename = os.path.splitext(os.path.basename(imgn))[0]
    #         f.writelines(basename + '\n')
    
    # with open('nguoc.txt', 'a') as f1:
    #     imgxuoi = sorted(glob.glob('{}/*'.format(des2)))
    #     for imgx in imgxuoi:
    #         basename = os.path.splitext(os.path.basename(imgx))[0]
    #         f1.writelines(basename + '\n')
    