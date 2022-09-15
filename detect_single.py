# from utils.datasets import *
# from utils.utils import *
import torch
import cv2
import numpy as np
import time
import random
import glob
import os
from tqdm import tqdm
import copy
import time
import torchvision

from utils.datasets import letterbox
from utils.general import non_max_suppression,xyxy2xywh,scale_coords

cuda = True
device = torch.device('cuda:0' if cuda else 'cpu')
SIZE = 640

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()
    
    
def get_model(weights):
    #fuse conv_bn and repvgg
    model = torch.load(weights, map_location=device)['model'].float().fuse().eval()
    #only fuse conv_bn
    #model = torch.load(weights, map_location=device)['model'].float().fuse().eval()
    return model
    
    
def process_imgs(orgimg_list):
    image_list = []
    for orgimg in orgimg_list:
        image = copy.deepcopy(orgimg)
        img = letterbox(image, new_shape=(SIZE,SIZE), auto=False)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image_list.append(img)
    imgs = np.array(image_list)
    imgs = np.ascontiguousarray(imgs)
    imgs = torch.from_numpy(imgs).to(device)
    imgs = imgs.float()  # uint8 to fp16/32
    imgs /= 255.0  # 0 - 255 to 0.0 - 1.0
        
    #if img.ndimension() == 3:
    #    img = img.unsqueeze(0)
    return imgs
    

def qiepian(image, num=3):
    if num == 2:   
        t_img = np.ones((640, 1066, 3), dtype=np.uint8)
        t_img[:, :, :] = 114
        s_image = copy.deepcopy(image)
        r_image = cv2.resize(s_image, (1066, 466))
        t_img[87:553, :, :] = r_image
        image1 = copy.deepcopy(t_img[:, 0:640, :])
        image2 = copy.deepcopy(t_img[:, 426:1066, :])
        return process_imgs([image1, image2])
    if num == 3:
        s_image = copy.deepcopy(image)
        r_image = cv2.resize(s_image, (1462, 640))
        image1 = copy.deepcopy(r_image[:, 0:640, :])
        image2 = copy.deepcopy(r_image[:, 411:1051, :])
        image3 = copy.deepcopy(r_image[:, 822:1462, :])
        return process_imgs([image1, image2, image3])
       
 

def process_img(orgimg):
    image = copy.deepcopy(orgimg)
    img = letterbox(image, new_shape=(SIZE,SIZE), auto=False)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img
    
    
def detect_qiepian(model, image, conf_thres, iou_thres):
    img = copy.deepcopy(image)
    split_num = 3
    detcount = 0
    start = time.time()
    inputs = qiepian(img, split_num)
    t1 = time.time()
    print('process img time: ', 1000*(t1 - start), ' ms ')
    n, c, h, w = inputs.shape
    s_h, s_w, s_c = image.shape
    pred = model(inputs)[0]
    t2 = time.time()
    print('model pred time: ', 1000*(t2 - t1), ' ms ')
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    gn = torch.tensor(image.shape)[[1, 0, 1, 0]]
    
    for i, det in enumerate(pred):
        if det is not None and len(det):
            if split_num == 3:
                det[:, :4] = scale_coords(inputs.shape[2:], det[:, :4], image.shape, ratio_pad=[[1462.0/s_w],[0, 0]]).round()
                if i == 1:
                    det[:, 0] = det[:, 0] + int(411 * 2048 / 1462.0)
                    det[:, 2] = det[:, 2] + int(411 * 2048 / 1462.0)
                if i == 2:
                    det[:, 0] = det[:, 0] + int(822 * 2048 / 1462.0)
                    det[:, 2] = det[:, 2] + int(822 * 2048 / 1462.0)
            if split_num == 2:
                det[:, :4] = scale_coords(inputs.shape[2:], det[:, :4], image.shape, ratio_pad=[[1066.0/s_w],[0, 87]]).round()
                if i == 1:
                    det[:, 0] = det[:, 0] + int(426 * 2048 / 1066.0)
                    det[:, 2] = det[:, 2] + int(426 * 2048 / 1066.0)
    #去掉边缘处的目标
    c = 10
    for i, det in enumerate(pred):
       if split_num == 3:
           if i == 0:
               a = int(640 * 2048 / 1462.0)
               pred[i] = det[det[:,2] < (a -c)]
           if i == 1:   
               a = int(411 * 2048 / 1462.0)
               b = int(1051 * 2048 / 1462.0)
               det = det[det[:,0] > (a + c)]
               pred[i] = det[det[:,2] < (b - c)]
           if i == 2: 
               a = int(822 * 2048 / 1462.0)
               pred[i] = det[det[:,0] > (a + c)]
       if split_num == 2:
           pass
           if i == 0:
               a = int(640 * 2048 / 1066.0)
               pred[i] = det[det[:,2] < (a -c)]
           if i == 1:   
               a = int(426 * 2048 / 1066.0)
               pred[i] = det[det[:,0] > (a + c)]
    
    if split_num == 3:
        all_pred = torch.cat((pred[0],pred[1],pred[2]), dim=0)
    if split_num == 2:
        all_pred = torch.cat((pred[0],pred[1]), dim=0)
        
    #切片图片的结果合并做nms
    boxes, scores = all_pred[:, :4] + all_pred[:, 5:6], all_pred[:, 4]
    i = torchvision.ops.nms(boxes, scores, iou_thres)
    nms_pred = all_pred[i]
    
    
    #print('nms_pred: ' , nms_pred.shape)
    #pred = non_max_suppression(pred, conf_thres, iou_thres)
    end = time.time()
    for *xyxy, conf, cls in nms_pred:
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        image = show_results(image, xywh, cls, conf)
        detcount += 1
        
    if split_num == 3:
        cv2.line(image, (int(411 * 2048 / 1462.0), 0), (int(411 * 2048 / 1462.0), 895), (0,255,0))
        cv2.line(image, (int(640 * 2048 / 1462.0), 0), (int(640 * 2048 / 1462.0), 895), (0,255,0))
        cv2.line(image, (int(822 * 2048 / 1462.0), 0), (int(822 * 2048 / 1462.0), 895), (0,255,0))
        cv2.line(image, (int(1051 * 2048 / 1462.0), 0), (int(1051 * 2048 / 1462.0), 895), (0,255,0))
    if split_num == 2:
        cv2.line(image, (int(426 * 2048 / 1066.0), 0), (int(426 * 2048 / 1066.0), 895), (0,255,0))
        cv2.line(image, (int(640 * 2048 / 1066.0), 0), (int(640 * 2048 / 1066.0), 895), (0,255,0))
    
    print('time: ', 1000*(end - start), ' ms ')
    return image, detcount


def show_results(img, xywh, class_num, conf=0.4):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    color = (0, 0, 255)
    cv2.rectangle(img, (x1,y1), (x2, y2), color, thickness=tl+2, lineType=cv2.LINE_AA)
    label = str(int(class_num)) + ' : ' + str(round(float(conf), 2))
    cv2.putText(img, label, (x1, y1 - 2), 0, tl , [225, 255, 255], thickness=tl+2, lineType=cv2.LINE_AA)
    return img


def detect(model, image, conf_thres, iou_thres):

    #img
    #h, w, c = image.shape
    #h_4, w_4 = h //4, w // 4
    #image = image[:, 240:1680, :]
    
    image = copy.deepcopy(image)
    img = process_img(image)
    #print(img.shape)
    start = time.time()
    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    end = time.time()
    print('time: ', 1000*(end - start), ' ms ')
    detcount = 0
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            # Write results
            for *xyxy, conf, cls in det:
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                image = show_results(image, xywh, cls, conf)
                detcount += 1
    return image, detcount




def detect_video(model, path, save_path = None):
    cv2.namedWindow("video",cv2.WINDOW_NORMAL)
    conf_thres, iou_thres = 0.5, 0.4
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #size = (1920, 1440)
    print('video fram size: ', size)
    save = False
    if save_path is not None:
        save = True
        print('save video to ', save_path)
        videoWriter = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)


    if capture.isOpened():
        index = 0
        while True:
            ret, frame = capture.read()
            if ret == True:
                frame, _ = detect_qiepian(model, frame, conf_thres, iou_thres)
                cv2.imshow('video', frame)
                cv2.imwrite('./test/tikou/images_n/' + str(index) + '.jpg', frame)
                index += 1
                if save:
                    videoWriter.write(frame)  # 写视频帧
            else:
                break
            if cv2.waitKey(1) == ord('q'):
                break
    cv2.destroyAllWindows()
    
    
def detect_image(model, path, save_path = None):
    conf_thres, iou_thres = 0.5, 0.3
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    frame = cv2.imread(path)
    frame, detcount = detect(model, frame, conf_thres, iou_thres)
    print('detcount: ', detcount)
    if save_path is not None:
        cv2.imwrite(save_path, frame)
    cv2.imshow('image', frame)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()


def detect_dir(model, path, save_dir):
    conf_thres, iou_thres = 0.5, 0.3
    files = sorted(glob.glob(os.path.join(path, '*.jpg')))
    for onefile in tqdm(files):
        frame = cv2.imread(onefile)
        frame, detcount = detect(model, frame, conf_thres, iou_thres)
        filepath,tempfilename = os.path.split(onefile)
        save_path = os.path.join(save_dir, tempfilename)
        if detcount == 0:
            cv2.imwrite(save_path, frame)


if __name__ == '__main__':
    #detect_test()
    weights = './test/gongbei/best.pt'
    model = get_model(weights)
    print('model: ', model)

    #video_path = '/home/xialuxi/work/python/PaddleDetection-release-2.3/20220510_20220510114006_20220510123906_114150.mp4'
    video_path = '/home/xialuxi/work/dukto/6月28日江苏梯口/1656280096084.mp4.h264'
    video_path1 = '/home/xialuxi/work/download/20220517_入境3号/128.0.3.26_8000_2_00157AE9A5A84A7BAC502EDD6CA4F6E5_/20220517_20220517102211_20220517141942_102222.mp4'
    video_path2 = '/home/xialuxi/work/download/20220517_入境3号/128.0.3.19_8000_2_8AE9D5C61D7847AAA31D875E1EDE9E4F_/20220517_20220517102205_20220517142803_102219.mp4'
    save_path = './test/save_gongbei_n.mp4'
    detect_video(model, video_path2, save_path)

    image_path = './test/youdu20220531191250.jpg'
    #detect_image(model, image_path)



    #detect_dir(model, './test/1', './test/result2')

