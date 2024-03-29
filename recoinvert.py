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
    img = cv2.resize(img, (640, 640))
    if img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = letterbox(img, 512)[0]
    img = img.transpose(2, 0, 1).astype(np.float32)
    img /= 255.0
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    img = img.cuda()
    #    model = torch.load(modelpath, map_location="cpu")["model"].float().eval()
    model = torch.load(modelpath, map_location="cpu")["model"].float().eval()
    model.cuda()
    with torch.no_grad():
        pre = model(img, augment=True)
    return pre


def poseprocess(pre, scorethresh):
    pre = pre[0]
    pre = non_max_suppression(pre)
    pre = pre[0]
    pre = pre.cpu().numpy()
    result = getjudge(pre, scorethresh)
    return result


def getjudge(pre, scorethresh):
    sumarea = 640*640
    mapdic = {0:"car", 1:"plate", 2:"hd", 3:"qd", 4:"fzcar"}
    fzcars = pre[pre[:,-1] == 4]
    cars = pre[pre[:, -1]==0]
    if fzcars.shape[0] == 0:
        return 0
    fzarea = (fzcars[:,2] - fzcars[:,0])*(fzcars[:,3] -fzcars[:,1])
    score = fzcars[np.argmax(fzarea)][-2]
    if fzarea[np.argmax(fzarea)] >= (sumarea/6) and score>=scorethresh:
        return 1
    else:
        return 0


def calculate(dirpath, scorethresh):
    matraix = np.zeros((2, 2))
    allcar = 1780
    allfanzcar = 1750
    modelpath = "/home/wujilong/code/yolov5_v2/yolov5-2.0/runs/fz_exp7/weights/best.pt"
    names = os.listdir(dirpath)
    count = len(names)
    carindex = 0
    feicarindex = 0
    thresh = 0
    for name in names:
        label = int(name.split('.')[0].split('_')[1])
        picpath = os.path.join(dirpath, name)
        try:
            pre = preprocess(picpath, modelpath)
            result = poseprocess(pre, scorethresh)
        except:
            continue

        if result == 0 and label==0:
            carindex += 1
        if result == 1 and label==1:
            feicarindex += 1
        thresh += 1
        if thresh%200==0:
            print("{}/{}".format(thresh, count))
    matraix[0][0] = carindex
    matraix[1][1] = feicarindex
    matraix[0][1] = allcar - carindex
    matraix[1][0] = allfanzcar - feicarindex
    print(matraix)
    print(scorethresh)
    print("car recall is: {}".format(matraix[0][0]/(matraix[0][0] + matraix[0][1])))
    print("car precision is:{}".format(matraix[0][0]/(matraix[0][0]+matraix[1][0])))
    print("fzcar recall is: {}".format(matraix[1][1] / (matraix[1][1] + matraix[1][0])))
    print("fzcar precision is:{}".format(matraix[1][1] / (matraix[1][1] + matraix[0][1])))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICE"] = "3"
    scores = [0.5, 0.6, 0.7, 0.8]
    dirpath = "/data/wujilong/tmpdata/labeldata/testfz"
    for score in scores:
        calculate(dirpath, score)

