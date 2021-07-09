import os
import random
import shutil

def getval(traindir):
#    labeldir = traindir.replace("images", "labels")
    tdir = os.path.join(traindir, "train")
    valdir = os.path.join(traindir, "val")
    os.makedirs(valdir, exist_ok=True)
    txtvaldir = valdir.replace("images", "labels")
    os.makedirs(txtvaldir, exist_ok=True)
    names = os.listdir(tdir)
    usenames = random.sample(names, 30)
    for usename in usenames:
        opath = os.path.join(tdir, usename)
        otxtpath = opath.replace("images", "labels")
        otxtpath = otxtpath.replace("jpg", "txt")
        txtname = usename.replace("jpg", "txt")
        valpath = os.path.join(valdir, usename)
        valtxtpath = os.path.join(txtvaldir, txtname)
        print(valtxtpath)
        shutil.move(opath, valpath)
        shutil.move(otxtpath, valtxtpath)

if __name__ == '__main__':
    traindir = "/home/wujilong/code/yolov5_v2/fzdata/images"
    getval(traindir)