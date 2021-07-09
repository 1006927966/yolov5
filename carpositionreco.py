from models.export import *
from utils.utils import *
import cv2


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.4, iou_thres=0.5, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def preprocess(path, modelpath):
    img = cv2.imread(path)
    img = cv2.resize(img, (640,640))
    if img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = letterbox(img, 512)[0]
    img = img.transpose(2, 0, 1).astype(np.float32)
    img /= 255.0
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    if img.ndimension()==3:
        img = img.unsqueeze(0)
    img = img.cuda()
#    model = torch.load(modelpath, map_location="cpu")["model"].float().eval()
    model = torch.load(modelpath, map_location="cpu").eval()
    model.cuda()
    with torch.no_grad():
        pre = model(img, augment=True)
    return pre


def poseprocess(pre):
    pre = pre[0]
    pre = non_max_suppression(pre)
    pre = pre[0]
    pre = pre.cpu().numpy()
    result = getjudge(pre)
    return result


def draw(img, pres, savepath):
    arrs = pres.numpy()
    for box in arrs:
        box=box[:4]
        leftpoint = (int(box[0]), int(box[1]))
        rightpoint = (int(box[2]), int(box[3]))
        cv2.rectangle(img, leftpoint, rightpoint, (0,255,0))
    cv2.imwrite(savepath, img)


def judgeinter(carbox, platebox):
    left = (max(carbox[0], platebox[0]), max(carbox[1], platebox[1]))
    right = (min(carbox[2], platebox[2]), min(carbox[3], platebox[3]))
    interare = (right[0] - left[0])*(right[1]-left[1])
    plateare = (platebox[2] - platebox[0])*(platebox[3]-platebox[1])
    if interare/plateare > 0.8:
        return True
    else:
        return False




def judgecarpos(judgemap):
    returnlist = [0]*9
    if judgemap["car"] == 0:
        returnlist[0] = 1
        return returnlist
    if judgemap["invert"]!=0:
        returnlist[8] = judgemap["car"]
    if judgemap["carplate"] == 0 :
        returnlist[7] = judgemap["car"]
        return returnlist
    if judgemap["platepos"]==0 and judgemap["light"] == 0:
        returnlist[1] = judgemap["car"]
        return returnlist
    if judgemap["platepos"]==1 and judgemap["light"] == 0:
        returnlist[2] = judgemap["car"]
        return returnlist
    if judgemap["platepos"]==2 and judgemap["light"] == 0:
        returnlist[3] = judgemap["car"]
        return returnlist
    if judgemap["platepos"]==2 and judgemap["light"] == 1:
        returnlist[4] = judgemap["car"]
        return returnlist
    if judgemap["platepos"]==0 and judgemap["light"] == 1:
        returnlist[5] = judgemap["car"]
        return returnlist
    if judgemap["platepos"]==1 and judgemap["light"] == 1:
        returnlist[6] = judgemap["car"]
        return returnlist



#car: [0,score], carplate:[0,1], platepose:[0(l), 1(r), 2(z)], light:[0"q",1"h"], invert:[0, 1]
def getjudge(arrs):
    sum_area = 640*640
    returnmap = ["wuche", "lefthead", "righthead", "head", "back", "leftback", "rightback", "side", "invert"]
    clsmap = {0:"car", 1:"carplate", 2:"hd", 3:"qd"}
    judgemap = {"car":0, "carplate":0, "platepos":0, "light":0, "invert":0}
    labels = [int(label) for label in arrs[:, -1]]
    cars = []
    plates = []
    qds = []
    hds = []
    for i in range(len(labels)):
        label = labels[i]
        if label == 0:
            car = [fac for fac in arrs[i]]
            car.append((arrs[i][2] - arrs[i][0])*(arrs[i][3] - arrs[i][1]))
            cars.append(car)
        elif label == 1:
            plate = [fac for fac in arrs[i]]
            plate.append((arrs[i][2] - arrs[i][0]) * (arrs[i][3] - arrs[i][1]))
            plates.append(plate)
        elif label == 2:
            hd = [fac for fac in arrs[i]]
            hd.append((arrs[i][2] - arrs[i][0]) * (arrs[i][3] - arrs[i][1]))
            hds.append(hd)
        else:
            qd = [fac for fac in arrs[i]]
            qd.append((arrs[i][2] - arrs[i][0]) * (arrs[i][3] - arrs[i][1]))
            qds.append(qd)
    cars = np.array(cars)
    qds = np.array(qds)
    hds = np.array(hds)
    plates = np.array(plates)
    if cars.shape[0] == 0 or cars[cars[:,-1]>(sum_area*1/6)].shape[0] == 0:
        returnlist = judgecarpos(judgemap)
        return np.array(returnlist)
    usefulcar = cars[np.argmax(cars[:, -1])]
    judgemap["car"] = usefulcar[-3]
    if plates.shape[0] != 0:
        maxplate = plates[np.argmax(plates[:, -1])]
        if judgeinter(usefulcar, maxplate):
            judgemap["carplate"] = 1
            maxcary = usefulcar[3]
            mincary = usefulcar[1]
            centery = (maxplate[1] +maxplate[3])/2
            if (maxcary-centery)/(centery-mincary) >2:
                judgemap["invert"] = 1
            carcenterx = (usefulcar[0] + usefulcar[2])/2
            platecenterx = (maxplate[0] + maxplate[2])/2
            platew = maxplate[2] - maxplate[0]
            if abs(platecenterx - carcenterx) < 0.5*platew:
                judgemap["platepos"] = 2
            if abs(platecenterx - carcenterx) >= 0.5*platew and platecenterx<carcenterx:
                judgemap["platepos"] = 0
            if abs(platecenterx - carcenterx) >= 0.5 * platew and platecenterx > carcenterx:
                judgemap["platepos"] = 1
    if qds.shape[0] == 0 and hds.shape[0]!=0:
        judgemap["light"] = 1
    if qds.shape[0] !=0 and hds.shape[0] ==0:
        judgemap["light"] =0
    if qds.shape[0] == 0 and hds.shape[0] ==0:
        judgemap["light"] =0
    if qds.shape[0]!=0 and hds.shape[0]!=0:
        maxqd = qds[np.argmax(qds[:, -1])]
        maxhd = hds[np.argmax(hds[:, -1])]
        if maxqd[-1] >= maxhd[-1]:
            judgemap["light"] = 0
            maxd = maxqd
        else:
            judgemap["light"] = 1
            maxd = maxhd
        dcentery = (maxd[1]+maxd[3])/2
        maxcary = usefulcar[3]
        mincary = usefulcar[1]
        if (maxcary - dcentery) / (dcentery - mincary) > 2:
            judgemap["invert"] = 1
    returnlist = judgecarpos(judgemap)
    return np.array(returnlist)


def testmatrix(savedir):
    dirpath = "/data/wujilong/tmpdata/labeldata"
    modelpath = "/home/wujilong/code/yolov5_v2/yolov5-2.0/runs/exp4/weights/best_o.pt"
    subnames = ["外观-左前", "外观-右前", "外观-正前","外观-正后","外观-左后","外观-右后", "外观-正侧"]
   # subnames = ["外观-左后","外观-右后"]
    matrix = np.zeros((len(subnames), len(subnames)))
    for i in range(len(subnames)):
        index = 0
        subname = subnames[i]
        print(subname)
        subdir = os.path.join(dirpath, subname)
        picnames = os.listdir(subdir)
        for picname in picnames:
            picpath = os.path.join(subdir, picname)
            try:
                pre = preprocess(picpath, modelpath)
                result = poseprocess(pre)
            except:
                print(picname)
                continue
            useresult = result[1:-1]
            if np.sum(useresult) > 1:
                print("Error :  this have two select!!!!!!!!")
            if np.sum(useresult) == 0:
                print("Error: get wuche !!!!!!!!")
                continue
            j = np.argmax(useresult)
            if j != i:
                savesubdir = os.path.join(savedir, subname)
                os.makedirs(savesubdir, exist_ok=True)
                savepicpath = os.path.join(savesubdir, picname)
                shutil.copy(picpath, savepicpath)
            matrix[i][j] += 1
            index += 1
            if index %100==0:
                print("{}/{}".format(index, len(picnames)))
    print(matrix)
    for p in range(len(subnames)):
        pt = matrix[p][p]
        pret = np.sum(matrix[:,p])
        ret = np.sum(matrix[p,:])
        recall = pt/ret
        precision = pt/pret
        print("{} recall is:{}".format(subnames[p], recall))
        print("{} precision is :{}".format(subnames[p], precision))
    











if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICE"] ="3"
    savedir = "/data/wujilong/tmpdata/errdata"
    testmatrix(savedir)
    # path = "/home/wujilong/code/yolov5_v2/yolov5-2.0/runs/exp4/weights/best.pt"
    # imgpathdir = "/data/wujilong/tmpdata/labeldata/外观-左前/"
    # cimg = cv2.imread(imgpath)
    # cimg = cv2.resize(cimg, (640,640))
    # # pre = preprocess(imgpath, path)
    # # print(pre[0].size())
    # savepath = "/home/wujilong/code/yolov5_v2/yolov5-2.0/runs/exp4/weights/best_o.pt"
    # pre = preprocess(imgpath, savepath)
    # pre = poseprocess(pre)
    # print(pre)
    # savedir = "/home/wujilong/code/yolov5_v2/tst.jpg"
    # draw(cimg, pre, savedir)
    print("ok")
   # model = torch.load(path, map_location="cpu")["model"].float().fuse()
   #  model = torch.load(savepath, map_location='cpu')
   #  print(model)
   #  print("ok")

