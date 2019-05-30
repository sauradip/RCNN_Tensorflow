import os
import glob
import shutil

DIR = '/media/buiduchanh/Work/Data/EKGIS/data_20190316/original_image'
img_des = '/media/buiduchanh/Work/Data/EKGIS/data_20190316/images'
json_des = '/media/buiduchanh/Work/Data/EKGIS/data_20190316/json'

jsonlist = sorted(glob.glob('{}/*.json'.format(DIR)))
for json in jsonlist:
    shutil.copy2(json, json_des)
    basename = os.path.splitext(os.path.basename(json))[0]
    imgname = basename + '.jpg'
    imgpath = os.path.join(DIR, imgname)
    shutil.copy2(imgpath, img_des)