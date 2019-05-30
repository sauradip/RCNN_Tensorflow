import os
import cv2
from libs.label_name_dict.label_dict import LABEl_NAME_MAP
from libs.configs import cfgs
import numpy as np
import cv2
import imutils
import json
import base64
import os
import glob
# boxes = np.array([[233.5084 ,  185.24924  , 45.020805, 141.2165  , -68.94732 ],
#  [281.85828 , 201.04012 ,  27.352657 , 16.700916, -69.017136],
#  [260.86395 , 194.19366  , 26.501448 , 18.458029 ,-72.80732 ],
#  [207.68018 , 172.18774 ,  27.73342 ,  21.53205 , -73.29729 ],
#  [245.24025 , 184.90349 ,  21.564772 , 20.781708 ,-69.50708 ],
#  [189.75484  ,166.22289 ,  24.714603 , 22.367182, -70.56879 ],
#  [226.7346 ,  181.56862 ,  23.640291 , 19.929073 ,-73.91228 ]]
# )

# category = [1., 2., 2., 3., 3., 7., 8.]

# basename = 'hinhanhdongho_1543878375904.jpg'
Des = cfgs.TEST_PATH

def make_json(img, rect, basename):

    imgpath = '/media/Hard_driver/R2CNN/metter/cs_data/images'
    jsonpath = '/media/Hard_driver/R2CNN/metter/cs_data/json'

    cv2.imwrite('{}/{}'.format(imgpath, basename), img)
    print('save img done')
    img = cv2.imread('{}/{}'.format(imgpath, basename))
    h, w, c = img.shape
    
    with open('{}/{}'.format(imgpath, basename), 'rb') as f:
        imageData = f.read()
        imageData = base64.b64encode(imageData).decode('utf-8')

    new_json = {}
    new_json["version"] = "3.11.2"
    new_json["flags"] = {}
    new_json["imageWidth"] = w
    new_json["imageHeight"] = h

    new_json["lineColor"] = [ 0, 255, 0, 128 ]
    new_json["imagePath"] = basename 
    new_json["fillColor"] = [255, 0, 0, 128]
    new_json["shapes"] = []
    new_json["imageData"] = imageData
    
    
    tmp = {}
    tmp["shape_type"] = 'polygon'
    tmp["line_color"] = None
    tmp["fill_color"] = None
    tmp["label"] = 'cs'

    tmp["points"] = np.array(rect).astype(int).tolist()
    new_json["shapes"].append(tmp)


    with open('{}/{}.json'.format(jsonpath, basename[:-4]), 'w') as outfile:  
        json.dump(new_json, outfile) 
    
    print('save json done')
def process(boxes, labels, img, basename, save_dir):

    # basename = basename[:-4]
    new_boxes = []
    new_label = []

    img = img + np.array(cfgs.PIXEL_MEAN)
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)
    img = np.array(img, np.float32)
    img = np.array(img*255/np.max(img), np.uint8)
    Rotate_again = False

    (heigth, width) = img.shape[:2]
    (cx, cy) = (width // 2, heigth // 2)

    total_theta = []

    for i, box in enumerate(boxes):
        x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]
        
        label = labels[i]
        if label == 1:
            rect = ((x_c, y_c), (w, h), theta)
            rect = cv2.boxPoints(rect)
            rect = np.int0(rect)
            # make_json(img , rect, basename)

        # if label != 1:
            total_theta.append(theta)
        else:
            return 

    theta = np.mean(np.array(total_theta))
    rotated = rotate_bound(img, theta)

    (new_height, new_width) = rotated.shape[:2]
    (new_cx, new_cy) = (new_width // 2, new_height // 2)

    for i, box in enumerate(boxes):
        x_c, y_c, w, h = box[0], box[1], box[2], box[3]
        
        label = labels[i]
        if label != 0:
            color = (255, 0, 0)
            rect = ((x_c, y_c), (w, h), theta)
            rect = cv2.boxPoints(rect)
            rect = np.int0(rect)
            # print(rect, type(rect))

            if label == 1:
                csrect = rotate_box(rect, cx, cy, heigth, width, theta)
                csrect = np.array(csrect).astype(int)

                new_label.append(label)
                new_boxes.append(csrect)
                # cv2.drawContours(rotated, [csrect], -1, (0,255,0), 2)
            else:

                newrect = rotate_box(rect, cx, cy, heigth, width, theta)
                newrect = np.array(newrect).astype(int)

                new_label.append(label)
                new_boxes.append(newrect)
                # cv2.drawContours(rotated, [newrect], -1, (0,255,0), 2)


    # cv2.imwrite(save_dir + '/' + basename , img)

    # return newboxex, newlabel, new_cx, new_cy, new_heghit, new_width
    new_boxes = np.array(new_boxes)
    new_label = np.array(new_label)
    

    for i, box in enumerate(new_boxes):
        label = new_label[i]
        if label == 1:
            distancex = np.amax(box[:,0]) - np.amin(box[:,0])
            distancey = np.amax(box[:,1]) - np.amin(box[:,1])
            
            if distancex < distancey : 
                Rotate_again = True
    
    if not Rotate_again:

        crop_img(rotated, new_boxes, new_label, Des, basename)

        cv2.imwrite(save_dir + '/' + basename.split('.')[0] + '_rotated.jpg' , rotated)

    if Rotate_again:
        rotated_again = rotate_bound(rotated, 90)
        tmp_box = []

        for i, box in enumerate(new_boxes):

            newrect_ = rotate_box(box, new_cx, new_cy, new_height, new_width, 90)
            newrect_ = np.array(newrect_).astype(int)
            tmp_box.append(newrect_)


        crop_img(rotated_again, tmp_box, new_label, Des, basename)

        cv2.imwrite(save_dir + '/' + basename.split('.')[0] + '_rotated.jpg' , rotated_again)

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def rotate_box(bb, cx, cy, h, w, theta):

    new_bb = list(bb)
    for i,coord in enumerate(bb):
        # opencv calculates standard transformation matrix
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
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

def crop_img(img, cnts, labels, des, basename):
    # img = cv2.imread("big_vertical_text.jpg")
    # cnt = np.array([
    #         [[64, 49]],
    #         [[122, 11]],
    #         [[391, 326]],
    #         [[308, 373]]
    #     ])
    for idx, cnt in enumerate(cnts):
        cnt = cnt.reshape(cnt.shape[0], 1, cnt.shape[1])
        print('idx', idx)
        label = labels[idx]
        category = LABEl_NAME_MAP[label]

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        print("rect: {}".format(box))
        # print("bounding box: {}".format(box))
        # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

        # img_crop will the cropped rectangle, img_rot is the rotated image
        img_crop, img_rot = crop_rect(img, rect)
        
        img_path = os.path.join(des, category)
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        h,w,c = img_crop.shape
        center = (w / 2, h / 2) 
        if h > w:   
            # rotated  = imutils.rotate(img_crop, 90) 
            rotated = imutils.rotate_bound(img_crop, 90)
            cv2.imwrite("{}/{}_{}.jpg".format(img_path, basename, idx), rotated)
            print("Save to {}/{}_{}.jpg".format(img_path, basename, idx))
        else:
            cv2.imwrite("{}/{}_{}.jpg".format(img_path, basename, idx), img_crop)
            print("Save to {}/{}_{}.jpg".format(img_path, basename, idx))
        

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
