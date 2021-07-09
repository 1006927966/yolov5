import cv2
import numpy as np
import os
import random
import shutil


def rotate(image, angle):
    (h,w) = image.shape[:2]
    (cx, cy) = (w/2, h/2)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    nw = int((h*sin) + (w*cos))
    nh = int((h*cos) + w*sin)
    M[0,2] += (nw/2) -cx
    M[1,2] += (nh/2) -cy
    image = cv2.warpAffine(image, M, (nw, nh))
    return image


def getfanz(dirpath):
    rootdir, _ = os.path.split(dirpath)
    savedir = os.path.join("/data/wujilong/tmpdata/labeldata/", "ztest")
    tsavedir = os.path.join(rootdir, 'z3', _)
    os.makedirs(savedir, exist_ok=True)
    names = os.listdir(dirpath)
    names = random.sample(names, 300)
    for name in names:
        savepath = os.path.join(savedir, name)
        print(savepath)
        picpath = os.path.join(dirpath, name)
        img = cv2.imread(picpath)
        #angle = random.randint(45, 135)
        angle = 0
        image = rotate(img, angle)
        cv2.imwrite(savepath, image)


def copyfz(cardir, fzdir, savedir):
    names = os.listdir(cardir)
    fznames = os.listdir(fzdir)
    for name in names:
        picpath = os.path.join(cardir, name)
        savename = name.split(".")[0] + "_0" + ".jpg"
        savepath = os.path.join(savedir, savename)
        shutil.move(picpath, savepath)

    for fzname in fznames:
        print(fzname)
        picpath = os.path.join(fzdir, fzname)
        savename = fzname.split(".")[0] + "_1" + ".jpg"
        savepath = os.path.join(savedir, savename)
        shutil.move(picpath, savepath)




if __name__ == "__main__":
    cardir = "/data/wujilong/tmpdata/labeldata/ztest"
    fzdir = "/data/wujilong/tmpdata/labeldata//fztest"
    savedir = "/data/wujilong/tmpdata/labeldata/testfz"
    os.makedirs(savedir, exist_ok=True)
    copyfz(cardir, fzdir, savedir)
    # rootdir = "/data/wujilong/二手车治理/车方位/二版方位数据"
    # #rootdir = "/data/wujilong/tmpdata/labeldata/"
    # subnames = ["外观-左前", "外观-右前", "外观-正前","外观-正后","外观-左后","外观-右后", "外观-正侧"]
    # for subname in subnames:
    #     dirpath = os.path.join(rootdir, subname)
    #     getfanz(dirpath)