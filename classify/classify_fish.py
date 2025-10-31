import numpy as np
import logging
import struct
import random
import base64
import json
import sys
import cv2
import os
import datetime
import math
import codecs
import csv
import time
import subprocess
import socket
import copy
import requests
import hashlib
import shutil
import timm

from logging.handlers import TimedRotatingFileHandler

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
#import pycls.core.builders as model_builder
#import pycls.core.builders as model_builder

from math import sqrt
from itertools import product as product
from torch.autograd import Variable, Function

#from pycls.core.config import cfg

import torch.backends.cudnn as cudnn
from numpy import random

from ultralytics import YOLO

## yolov6
# from yolov6.utils.events import load_yaml
# from yolov6.core.inferer import Inferer
# from yolov6.data.datasets import LoadData
# from yolov6.layers.common import DetectBackend
# from yolov6.utils.nms import non_max_suppression


torch.set_num_threads(1)

area={}
isoareaidmapping={}

# 读取配置
processnum=1
processid=0
detectionEnabled =True
use_gpu=0


zerobits=[0]*8

skipqqimagecheck=False

home='/home/hero/dongyu/classify/'

objects = ['person', 'bird', 'mammals', 'reptile', 'amphibia', 'fish', 'shrimp']

donotChkSources=set(['hx','an','tx','web'])

TMPDIR=home+"data/tmp"
RESULTDIR=home+"data/result"
PICDIR=home+"data/upload"
DESTDIR=home+"data/done"
PROCESSINGDIR=home+"data/processing"
CONTROLDIR=home+"data/control/"
CONFIGDIR=home+"data/conf/"
CHECKDIR="/home/hero/niaodian/check/"
LOGDIR=home+"data/log/"

DIRFOR87014='/home/hero/niaodian/classify/87014/'

serverurl = 'http://ca.dongniao.net/dypics/'

returnallresult=False

os.umask(0)

print("home: ", home, "LOGDIR: ", LOGDIR)

if not os.path.isdir(TMPDIR):
    try:
        os.mkdirs(TMPDIR)
        os.mkdirs(RESULTDIR)
        os.mkdirs(PICDIR)
        os.mkdirs(DESTDIR)
        os.mkdirs(PROCESSINGDIR)
        os.mkdirs(CONTROLDIR)
        os.mkdirs(CONFIGDIR)
        os.mkdirs(LOGDIR)
    except:
        pass

configfile=home+"data/conf/dongyu_daemon.conf"

fishwithbitsfile=home+"fishes_withbits.json"

# load fishwithbitsfile
fishwithbits=[]
if os.path.isfile(fishwithbitsfile):
    with open(fishwithbitsfile) as json_file:
        fishwithbits = json.load(json_file)


print("FM Configfile: " +  configfile)

configfp = codecs.open(configfile, encoding='utf-8', mode="r")
for line in configfp:
    arr = line.replace("\n", "").split("=")
    if arr[0]=='processnum':
        processnum=int(arr[1])
        if processnum<1 or processnum>4:
            print("porcessnum error: " + arr[1])
            quit()
    if arr[0]=="gpu":
        use_gpu=int(arr[1])

    if arr[0]=='skipqqimagecheck':
        if arr[1].upper()=='TRUE':
            skipqqimagecheck=True
        else:
            skipqqimagecheck=False

    # serverurl, for qq image check，对方会访问这个地址获取图片
    if arr[0]=='serverurl':
        serverurl=arr[1]
    
configfp.close()


# intialized the domain socket
socket_address = '/home/hero/dongyu/tmp/fish.socket'

# Make sure the socket does not already exist
try:
    os.unlink(socket_address)
except OSError:
    if os.path.exists(socket_address):
        raise

# Create a UDS socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

# Bind the socket to the address
print('starting up on {}'.format(socket_address))
sock.bind(socket_address)

os.chmod(socket_address,0o777)

# Listen for incoming connections
sock.listen(1)
sock.settimeout(0.5)

logid=1
gpuid=0
childrenPids={os.getpid()}
for i in range(processnum):
    if i==0:
        # parent
        continue
    print("forking: ", i)
    pid=os.fork()
    if pid==0:
        # child process
        logid += i
        if use_gpu>0:
            gpuid = (logid-1)%use_gpu
        break
    else:
        #parent
        childrenPids.add(pid)
        continue

mypid=str(os.getpid())


myformatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y%m%d%H%M%S ')
    
logger = logging.getLogger("Rotating Log")
logger.setLevel(logging.INFO)

logfilename=home+ 'data/log/classify.log_'+str(logid)

class InfoFilter(logging.Filter):
    def __init__(self, param):
        super().__init__()
        self.param = param

    def filter(self, record):
        return self.param in record.msg

handler = TimedRotatingFileHandler(logfilename,
                           when="midnight",
                           interval=1,
                           delay=False,
                           backupCount=5)
handler.setFormatter(myformatter)
handler.addFilter(InfoFilter("FM "))
logger.addHandler(handler)

if logid==1:
    logger.info("FM gpu: "+ str(use_gpu) + " procnum: " + str(processnum) )

if use_gpu>0:
    pidstr=str(logid)+"-g"+str(gpuid)+str("-")+mypid
    logger.info("FM "+pidstr + " Process " + str(logid) + "/" + str(processnum) + " GPU ID: " + str(gpuid) + " workingdir: " + home + " detection: " + ("enabled" if detectionEnabled  else "disabled") + " gpu: " + str(use_gpu) + " serverurl: " + serverurl)
    print("FM "+pidstr + " Process " + str(logid) + "/" + str(processnum) + " GPU ID: " + str(gpuid) + " workingdir: " + home + " detection: " + ("enabled" if detectionEnabled  else "disabled") + " gpu: " + str(use_gpu)+ " serverurl: " + serverurl) 
else:
    pidstr=str(logid)+"-c-"+mypid
    logger.info("FM "+pidstr + " Process " + str(logid) + "/" + str(processnum) + " Using CPU  workingdir: " + home + " detection: " + ("enabled" if detectionEnabled  else "disabled")  + " serverurl: " + serverurl)
    print("FM "+pidstr + " Process " + str(logid) + "/" + str(processnum) + " Using CPU  workingdir: " + home + " detection: " + ("enabled" if detectionEnabled  else "disabled") + " serverurl: " + serverurl)


print("process: "+pidstr)


IMAGE_SHAPE = (300, 300)

if sys.version[:1]!='3':
    print("For version 3 only.\n")
    quit()


class_pb=home+'mymodels/fish_cls_1dd43a659e87dc4fcd4eef5da7dc5e4a62998df8.pth'
class_pb=home+'mymodels/fish_20251029_86.21.pth'
labelmapfile=home+'mymodels/fish_20251029_86.21.pth.labelmap.csv'


# YOLOModelFile=home+'mymodels/yolo8s.20231024.pt'
# YOLOModelFile=home+'mymodels/yolo8m.20231026.pt'
# YOLOModelFile=home+'mymodels/yolo8m.20231029.pt'

# YOLOModelFile=home+'mymodels/yolo8m.20240104.800.pt'

# RAMBP = Reptile Amphibian Mammal Bird Person Fish Shrimps
YOLOModelFile=home+'mymodels/yolo8m.20241002.640.6930.pt'

logging.getLogger().setLevel(logging.WARNING)

#SCORE_THRESHOLD = 0.38
SCORE_THRESHOLD = 0.38
#SCORE_THRESHOLD = 0.12

IOU_THRES = 0.6

Cropped_extention = 0.03

weixintoken=""

# 加载 fishes
fishes = []
fishes_dict = {}
fp = codecs.open("../final.fish.2025.txt", encoding='utf-8', mode="r")
for line in fp:
    arr = line.replace("\n", "").split(",")
    fishes.append(arr)
    fishes_dict[int(arr[0])] = {
        'sn': arr[1],
        'en': arr[2],
        'phylum': arr[3],
        'class': arr[4],
        'order': arr[5],
        'family': arr[6],
        'status': arr[7],
        'inatId': arr[8],
        'wormsId': arr[9],
        'genus': arr[1].split(" ")[0],
    }
print("Fish species #: " + str(len(fishes)))

# 加载 lables
labelmap = {}
fp = codecs.open(labelmapfile, encoding='utf-8', mode="r")
for line in fp:
    arr = line.replace("\n", "").split(",")
    labelmap[int(arr[0])]=arr[1]

# 加载 taxons
taxons = {}
fp = codecs.open("taxon_names_fish.json", encoding='utf-8', mode="r")
taxons=json.load(fp)
fp.close()

lastday=""


def getDomainName(url):
    return url.split("//")[-1].split("/")[0].split('?')[0]

def tencentImageCheck(serverurl, targetdir,birdpic,openid,weixintoken):

    command="/usr/bin/curl -s -m 5 -d '{ \"media_url\":\"" + serverurl + targetdir + "/" + birdpic+"\",\"media_type\":2,\"version\":2,\"scene\":1,\"openid\":\""+ openid +"\"}' 'https://api.weixin.qq.com/wxa/media_check_async?access_token=" +weixintoken+"'"
    runningproc= subprocess.Popen(command, shell=True, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("FM "+datetime.datetime.now().strftime("%Y%m%d %H%M%S ")+ str(time.time()-t1)+ " qq check: " + command)
    sys.stdout.flush()

    return runningproc


def getRiskyFromQQResult(traceid):
    checkfilename=CHECKDIR+traceid+".qqresult"
    count=0
    while not os.path.isfile(checkfilename):
        if count>12:
            # 10秒后还没有结果，返回超时
            print("FM "+datetime.datetime.now().strftime("%Y%m%d %H%M%S ")+ "getRiskyFromQQResult: ", traceid, "timeout")
            return "?"
        count+=1
        time.sleep(1)
        continue
    with open(checkfilename) as fp:
        isrisky=fp.read().replace("\n","")
    fp.close()
    print("FM "+datetime.datetime.now().strftime("%Y%m%d %H%M%S ")+ "getRiskyFromQQResult: ", traceid, isrisky)
    sys.stdout.flush()
    return isrisky

def checkQQResult(runningproc, birdpic):
    qqstarttime=time.time()
    print("FM "+datetime.datetime.now().strftime("%Y%m%d %H%M%S ") + " *** check qq img result started")
    sys.stdout.flush()
    while True:
        retcode = runningproc.poll()
        if retcode is not None: # Process finished.
            break
        else: # No process is done, wait a bit and check again.
            time.sleep(.1)
            continue
    resultfromweixin=runningproc.stdout.readline()
    try:
        checkresult=json.loads(resultfromweixin)
        if "trace_id" in checkresult:
            traceid=checkresult["trace_id"]
            domainname=getDomainName(serverurl)
            if domainname!='dongniao.net':
                command = "curl -s -m 5 'https://dongniao.net/cs?fromotherserver="+domainname+"&trace_id="+traceid+"'"
                result=subprocess.run(command,capture_output=True, text=True, shell=True)
                print("FM "+datetime.datetime.now().strftime("%Y%m%d %H%M%S ") + " *** tell server to redicredt: ", command, result.stdout)
                sys.stdout.flush()
                if result.stdout.isdigit():
                    isrisky=result.stdout
                else:
                    isrisky=getRiskyFromQQResult(traceid)
            else:
                isrisky=getRiskyFromQQResult(traceid)
            print("FM "+datetime.datetime.now().strftime("%Y%m%d %H%M%S ") + str(round(time.time()-qqstarttime,3)) + " *** check qq img result ended: ", birdpic, traceid, isrisky)
            sys.stdout.flush()
            return isrisky,traceid
    except Exception as e:
        print("QQ CHECK Error: ", e)
        print("QQ CHECK Error: ", birdpic)
        sys.stdout.flush()
    
    return "?",""

def IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    if boxA[5]!=boxB[5]:
        return 0
    xA = max(boxA[1], boxB[1])
    yA = max(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[4], boxB[4])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[3] - boxA[1] + 1) * (boxA[4] - boxA[2] + 1)
    boxBArea = (boxB[3] - boxB[1] + 1) * (boxB[4] - boxB[2] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def inference(file, model):
    results = model(file,conf=SCORE_THRESHOLD,imgsz=640,device='cpu')
    boxes = []
    for r in results:
        b=(r.boxes)  # print the Boxes object containing the detection bounding boxes
        confs = b.conf
        cls = b.cls
        xyxy = b.xyxy
        # print("    r----> ", r.names)
        # print("    b----> ", b)
        # print("    cls----> ", b.cls, r.names)

        myboxes = []
        for i in range(len(xyxy)):
            myboxes.append([float(confs[i]), int(xyxy[i][0]), int(xyxy[i][1]), int(xyxy[i][2]), int(xyxy[i][3]), r.names[int(cls[i])]])

        for i in range(len(myboxes)):
            # 所有类别全部返回
            # if cls[i] != 2:
            #     continue
            curr = myboxes[i]
            overlap = False
            for box in boxes:
                if curr[5] != box[5]:
                    continue
                iou = IOU(box, curr)
                if iou > IOU_THRES:
                    overlap = True
                    break
            if overlap:
                continue
            boxes.append(curr)

        return boxes 

def readWxTokenFile(wxtokenfile):
    weixintoken=""
    if os.path.isfile(wxtokenfile):
        tokenfp = codecs.open(wxtokenfile, encoding='utf-8', mode="r")
        for line in tokenfp:
            weixintoken = line.replace("\n", "")
            break
        tokenfp.close()
        print("FM "+datetime.datetime.now().strftime("%Y%m%d %H%M%S ")+ " renewed weixin token from " +wxtokenfile + " : " + weixintoken)
    else:
        print("FM "+datetime.datetime.now().strftime("%Y%m%d %H%M%S ")+ " failed to renew weixin token")
    sys.stdout.flush()
    return weixintoken

def getweixintoken(lasthour):
    currenthour=datetime.datetime.now().strftime("%Y%m%d%H")
    dyweixintoken=""
    if currenthour != lasthour:
        dyweixintoken = readWxTokenFile("/home/hero/dongniao/weixin/"+currenthour+".dy.wxtoken")
    return currenthour, dyweixintoken

def mySleep():
    # print("FM "+datetime.datetime.now().strftime("%Y%m%d %H%M%S ")+ 'waiting for a connection')
    try:
        connection, _ = sock.accept()
    except socket.timeout:
        #print("timeout")
        pass
        return
    
    try:
        # print('wakeup by connection')
        # Receive the data in small chunks and retransmit it
        while True:
            data = connection.recv(64)
            if not data:
                break
    finally:
        # Clean up the connection
        connection.close()
    return

def generate_nonFishResult(boxes, height, width):
    resp3=[]
    for i in range(len(boxes)):
        box=[0,0,width,height]
        try:
            box=[int(math.floor(float(boxes[i][1]))),int(math.floor(float(boxes[i][2]))),int(math.ceil(float(boxes[i][3]))),int(math.ceil(float(boxes[i][4])))]
        except:
            box=[0,0,width,height]
        resultlist=[]
        result=[]
        result.append(round(float(boxes[i][0]*100),2))
        result.append(boxes[i][5])
        result.append(-1)
        resultlist.append(result)
        a = {}
        a['box']=box
        a['list']=resultlist
        resp3.append(a)    
    return resp3

def loadAreaMapping():
    global area,isoareaidmapping
    totalSpcies=0
    totalRecord=0
    fp=open("alphacodeareaidmapping.csv")
    for line in fp:
        arr = line.replace("\n", "").split(",")
        isoareaidmapping[arr[0].upper()]=int(arr[1])
    fp.close
    fp=open("distributionallfordaemon.csv")
    for line in fp:
        arr = line.replace("\n", "").split(",")
        species=set([int(x) for x in arr[3:]])
        area[int(arr[0])]=[arr[1],arr[2],species]
        totalSpcies+=len(arr)-3
        totalRecord+=1
    print("Area records loaded: ", totalRecord, totalSpcies )
    fp.close
    return

def fishinarea(birdid, areaid):
    # 目前不处理任何 areaid，直接返回真
    return True
    global area
    # print ("birds in area: ",birdid, birdid in area[areaid][2])
    return areaid in area and birdid in area[areaid][2]

def getareaid(areacode):
    if areacode in isoareaidmapping:
        areaid=isoareaidmapping[areacode]
    else:
        print("FM "+datetime.datetime.now().strftime("%Y%m%d %H%M%S ")+ pidstr + " "+ 'areacode error, ignored: ',areacode, birdpic)
        sys.stdout.flush()
        areaid=-2
    return areaid

def cubeTheImage(img, a1, b1, a2, b2, width, height):

    if a1<0:
        a1=0
    if b1<0:
        b1=0
    if a2>a1+width:
        a2=a1+width
    if b2>b1+height:
        b2=b1+height

    leftpad=0
    rightpad=0
    toppad=0
    bottompad=0

    if b2-b1>a2-a1:
        leftpad=int((b2-b1- (a2-a1))/2)
        rightpad=(b2-b1- (a2-a1))-leftpad
    else:
        toppad = int((a2-a1-(b2-b1))/2)
        bottompad=(a2-a1-(b2-b1))-toppad

    my_cropped_image = img[int(math.floor(b1)):int(math.ceil(b2)), int(math.floor(a1)):int(math.ceil(a2))]
    # print("cropping2: ",int(math.floor(b1)),int(math.ceil(b2)), int(math.floor(a1)),int(math.ceil(a2)))
    
    my_squared_image = cv2.copyMakeBorder(my_cropped_image,int(toppad),int(bottompad),int(leftpad),int(rightpad),cv2.BORDER_CONSTANT,value=[0,0,0])

    return my_squared_image

nonFishList={
    -1: 'bird',
    -2: 'mammals',
    -3: 'reptile',
    -4: 'amphibia',
    -5: 'others',
}

def getSpeciesInfoFromId(id):
    if id in labelmap:
        sid=labelmap[id]
        classid=sid[:1]
        rid=int(sid[1:])
        if classid=='F' or classid=='S':
            genus=fishes_dict[rid]['genus']
            # print("genus: ",genus)
            genuslower=genus.lower()
            family=fishes_dict[rid]['family']
            familylower=family.lower()
            taxonString=(taxons[genuslower][2]+"|" + taxons[genuslower][3] + "|" + taxons[genuslower][1]) if genuslower in taxons else "||"+genus
            taxonString+="|" + (taxons[familylower][2]+"|" + taxons[familylower][3] + "|" + taxons[familylower][1]) if familylower in taxons else "||"+family

            return classid,rid,fishes[rid][3] + "|" + fishes[rid][2] + "|" + fishes[rid][1]+"|"+taxonString, fishwithbits[rid][3:] if rid<len(fishwithbits) else zerobits 
        else:
            return classid,rid,fishes[rid][3] + "|" + fishes[rid][2] + "|" + fishes[rid][1], fishwithbits[rid][3:] if rid<len(fishwithbits) else zerobits
    else:
        return "",-1,"",zerobits


def generate_results(res, x1,y1,x2,y2,height, width,areaid=-1, boxconfidence=1.0):

    # 目前版本不处理任何 areaid
    areaid=-1

    resp3 = []
    
    box=[0,0,width,height]
    try:
        box=[int(math.floor(float(x1))),int(math.floor(float(y1))),int(math.ceil(float(x2))),int(math.ceil(float(y2)))]
    except:
        box=[0,0,width,height]

    resultlist=[]
    
    localresult=[]
    pos=0
    totalScore=0
    degrade=[]
    
    for id, accu in res:

        score=round(float(accu*100),2)
        if pos==0:
            firstScore=score

        if id>=0:
            filtered= (areaid>=0 and (not fishinarea(id, areaid)) )  # or (id in extinct)
            # 当中文名为空时返回英文名，兽类
            classid, speciesid, name, bits = getSpeciesInfoFromId(id)
            
            result=[score, name, speciesid, classid, bits]
            
        else:
            filtered = False
            result=[score,nonFishList[int(id)],int(id),"-", zerobits]
        
        resultlist.append(result)
        
        if not filtered and (score>2 and score>0.09*firstScore or len(localresult)==0 or returnallresult):
            #degrade
            if pos==0: 
                degrade.append(1.0)
            else:
                degrade.append(((score+(firstScore-score)*2/3)/firstScore))

            totalScore+=score
            localresult.append(copy.deepcopy(result))
        pos+=1

    if len(localresult)<len(resultlist): # 重新计算 score for localresult
        rate = 1+len(localresult)/len(resultlist)
        for i in range(len(localresult)):
            newscore=round(localresult[i][0] / totalScore * 1000 / math.sqrt(math.sqrt(rate))* degrade[i]) / 10
            localresult[i][0]=newscore

    a = {}
    a['box']=box
    
    a['list']=localresult

    a['boxconf']=round(boxconfidence*100,2)

    return a


def get_square_coordinates(box, height, width):
    
    widthextention = (box[3]-box[1])* Cropped_extention 
    heightextention = (box[4]-box[2])* Cropped_extention 

    xmin = box[1] - widthextention / 2
    ymin = box[2] - heightextention / 2
    xmax = box[3] + widthextention / 2
    ymax = box[4] + heightextention / 2

    if xmin<0:
        xmin=0
    if ymin<0:
        ymin=0
    if xmax>width:
        xmax=width
    if ymax>height:
        ymax=height

    return int(ymin), int(xmin), int(ymax), int(xmax)


def predict(img,net,softmax):
    t0 = time.time()

    img = cv2.resize(img, (300, 300))
    tensor_img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float()/255.0

    tmp1=time.time()
    result = net(tensor_img)
    tmp2=time.time()
    # logger.info("FM "+pidstr + " "+'{:1.3f}'.format(tmp2-tmp1)+" calling classify models done" + url)

    values, indices = torch.topk(result, 10)
    values = softmax(values)
    t1 = time.time()
    return zip(indices[0].tolist(), values[0].tolist())

def precess_image(img_src, img_size, stride, half):
    '''Process image before image inference.'''
    image = letterbox(img_src, img_size, stride=stride)[0]
    # Convert
    image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.half() if half else image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0

    return image, img_src

def rescale(ori_shape, boxes, target_shape):
    '''Rescale the output to the original image shape'''
    ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
    padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

    boxes[:, [0, 2]] -= padding[0]
    boxes[:, [1, 3]] -= padding[1]
    boxes[:, :4] /= ratio

    boxes[:, 0].clamp_(0, target_shape[1])  # x1
    boxes[:, 1].clamp_(0, target_shape[0])  # y1
    boxes[:, 2].clamp_(0, target_shape[1])  # x2
    boxes[:, 3].clamp_(0, target_shape[0])  # y2

    return boxes


def box_fully_contains(b1,b2):
    #print("contains: ", b1,b2)
    if b1[0]<=b2[0]+1 and  b1[1]<=b2[1]+1 and b1[2]>=b2[2]-1 and b1[3]>=b2[3]-1:
        return True
    elif b1[0]>=b2[0]-1 and  b1[1]>=b2[1]-1 and b1[2]<=b2[2]+1 and b1[3]<=b2[3]+1:
        return True
    else:
        return False

def getkey(item):
    if len(item['list'])>0:
        return item['list'][0][0]
    else:
        return 0

def cleanupResult(results):
    if len(results)<2:
        return results
    r = sorted(results, key=getkey, reverse=True)
    # print("----->",r)
    # 如果有超过一人列表，删除 Top1 小于 0.2 的 box
    while len(r)>1 and (len(r[len(r)-1]['list'])==0 or r[len(r)-1]['list'][0][0]<20.0):
        r.pop()

    rr=[]
    rr.append(r[0])
    for i in range(1,len(r)):
        fullyCotains=False
        for j in range(len(rr)):
            #print(i, r[i]['box'], j, rr[j]['box'])
            if box_fully_contains(r[i]['box'],rr[j]['box']):
                fullyCotains=True
                break
        if not fullyCotains:
            rr.append(r[i])
    return rr

def checkQuit():
    for control in os.listdir(CONTROLDIR):
        if control==("all.stop") and logid==1:
            # 父亲
            for pid in childrenPids:
                controlfp=open(CONTROLDIR+str(pid)+".stop",  "wb", buffering=0)
                controlfp.close()
            os.remove(CONTROLDIR+control)
            return
        if control.endswith(".stop") and control.split('.')[0]==mypid:
            sys.stdout.flush()
            os.remove(CONTROLDIR+control)
            print("FM "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S ")+ mypid + " QUIT.")
            logger.info("API daemon " + str(logid) + " "+ mypid + " Quit")
            quit()


if __name__ == '__main__':
    print("FM "+datetime.datetime.now().strftime("%Y%m%d %H%M%S ")+ "Service started.")

    # loadAreaMapping()

    # Initialize YOLO model
    if use_gpu > 0:
        gpucpu = str(gpuid)
    else:
        gpucpu = 'cpu'

    if detectionEnabled :
        model = YOLO(YOLOModelFile)

    print("detection model loaded: " + YOLOModelFile)

    # net = timm.create_model("convnextv2_tiny", num_classes=16800)
    net = timm.create_model("convnextv2_tiny", num_classes=21650)
    net = torch.compile(net, dynamic=False)
    state_dict = torch.load(class_pb, map_location="cpu")
    del state_dict["_config"]
    net.load_state_dict(state_dict)
    net.eval()
    net = net.float()
    softmax = nn.Softmax(dim=1).eval()

    #net = EfficientNet.from_name('efficientnet-b2', override_params={'num_classes': 11000})

    print("classification model loaded: " + class_pb)
    sys.stdout.flush()

    if not skipqqimagecheck:
        lasthour, dyweixintoken = getweixintoken("")

    t1=t2=t3=t4=0

    while 1:

        t=0

        res=[]

        for birdpic in os.listdir(PICDIR):

            resp3 = []

            areaid=-1

            if birdpic.endswith(".jpg") or birdpic.endswith(".url"):

                originalfile=PROCESSINGDIR+"/"+pidstr+"."+birdpic

                try:
                    os.rename(PICDIR+"/"+birdpic, originalfile)
                except:
                    continue

                if not os.path.isfile(originalfile):
                    continue

                print("FM "+datetime.datetime.now().strftime("%Y%m%d %H%M%S ")+ pidstr + " "+ 'Todo found: ',birdpic)
                sys.stdout.flush()

                t1=time.time()

                picid = birdpic[:-10]

                url=""
                tuyapass=""
                if birdpic.endswith('.url'):
                    urlfp=open(originalfile)
                    i=0
                    for line in urlfp:
                        if i==0:
                            url=line.replace("\n","")
                        elif i==1:
                            tuyapass=line.replace("\n","")
                        else:
                            break
                        i+=1
                    urlfp.close

                    jpgfilename=".jpg".join((PROCESSINGDIR+"/"+pidstr+"."+birdpic).rsplit(".url", 1)) # replace the last ".url" with ".jpg"
                    t6=time.time()

                    if tuyapass!="":
                        print(birdpic, "Tuya OSS file...", tuyapass, url)
                        try:
                            TuyaDecrypt.decrypt_oss_file(url,tuyapass,jpgfilename)
                        except requests.exceptions.ReadTimeout:
                            print("OSS file fetching timeout")
                            pass 
                        except:
                            print("OSS file fetching error")
                            pass 
                        command="Tuya jpg fetched."                
                    else:
                        command="/usr/bin/curl -s -m 5 -L --max-redirs 2 -o "+ jpgfilename + " '"+url+"'"
                        print("command", command)
                        os.system(command)
                    t7=time.time()

                    print("FM "+datetime.datetime.now().strftime("%Y%m%d %H%M%S ")+ pidstr + " "+ 'Url fetching time: '+str(round((t7-t6)*1000))+"ms", command)
                    sys.stdout.flush()

                    urlsignalfile=originalfile
                    originalfile=jpgfilename

                    

                if not birdpic.endswith(".cubed.jpg"):
                    picid = birdpic[:-4]

                splits=picid.split('_')

                serverid='n'
                dnid='nobody'
                openid='nobody'

                if len(splits)>=7:
                    #new version
                    a1=splits[3]
                    b1=splits[4]
                    a2=splits[5]
                    b2=splits[6]
                    try:
                        a1=int(a1)
                        b1=int(b1)
                        a2=int(a2)
                        b2=int(b2)
                    except:
                        a1=0
                        b1=0
                        a2=width
                        b2=height

                    if len(splits)>7:
                        serverid=splits[7]

                    areaid=-1

                    if len(splits)>8:
                        if picid.startswith("dy"):
                            openidanddnid=splits[8].split("-")
                            try:
                                openid=bytes.fromhex(openidanddnid[0]).decode('utf-8').replace('"','').replace("\\","")
                            except:
                                openid=openidanddnid[0]
                            if len(openidanddnid)>1:
                                dnid=openidanddnid[1]
                            if len(splits)>9:
                                areaid=getareaid(splits[9])
                        else:
                            areaid=getareaid(splits[8])
                else:
                    if len(splits)>3:
                        serverid=splits[3]

                    areaid=-1
                    md5digest=""
                    if picid.startswith("dy"):
                        if len(splits)>4:
                            md5digest=splits[3]
                            serverid=splits[4]

                        if len(splits)>5:
                            openidanddnid=splits[5].split("-")
                            try:
                                openid=bytes.fromhex(openidanddnid[0]).decode('utf-8').replace('"','').replace("\\","")
                            except:
                                openid=openidanddnid[0]
                            if len(openidanddnid)>1:
                                dnid=openidanddnid[1]

                        
                        if len(splits)>6:
                            areaid=getareaid(splits[6])
                    else:
                        if len(splits)>3:
                            serverid=splits[3]

                        if len(splits)>4:
                            if not picid.startswith("ugc"):
                                areaid=getareaid(splits[4])
                            else:
                                try:
                                    areaid=int(splits[4])
                                except:
                                    areaid=25600

                imgsource=splits[0]

                loggingTag="FM "

                targetdir=picid.split('_')[1]
                
                # 早一些放入目标目录，因需要尽快异步调用 QQ imgcheck

                if not os.path.isdir(DESTDIR+"/"+targetdir):
                    os.mkdir(DESTDIR+"/"+targetdir,0o777)

                try:
                    os.rename(PROCESSINGDIR+"/"+pidstr+"."+birdpic, DESTDIR+"/"+targetdir+"/"+birdpic)
                except:
                    pass

                originalfile=DESTDIR+"/"+targetdir+"/"+birdpic

                if not skipqqimagecheck and not imgsource in donotChkSources :
                    # 异步调用 QQ imgcheck 
                    # command="/usr/bin/curl -s -m 5 -F media=@"+ originalfile + " 'https://api.weixin.qq.com/wxa/img_sec_check?access_token="+weixintoken+"'" # version 1, depreciated.
                    runningproc = tencentImageCheck(serverurl, targetdir,birdpic,openid,dyweixintoken)
  

                t2 = time.time()

                height=0
                width=0
                try:
                    raw_image = cv2.imread(originalfile)

                    if raw_image is None:
                        print("FM "+datetime.datetime.now().strftime("%Y%m%d %H%M%S ")+ ('Image file {} does not exist'.format(originalfile)))

                    height, width = raw_image.shape[:2]
                except:
                    if url!="":
                        resp3=["Url"]
                    else:
                        resp3=["Format"]
                    height=0
                    width=0
            
                t3=time.time()
                td1=t3
                td2=t3

                yolonobox=False
                
                needtocheckqq=True
                boxes=[]

                if width>0 and height>0:

                    birdcount=1

                    boxdetected=False

                    if len(splits)>=7:
                        #new version

                        if areaid>=-1:

                            cropped = cubeTheImage(raw_image, a1, b1, a2, b2, width, height)

                            t3=time.time()

                            print("FM "+datetime.datetime.now().strftime("%Y%m%d %H%M%S ")+ pidstr + " "+ 'start classification: ',picid)
                            sys.stdout.flush()

                            predictResult = predict(cropped, net,softmax)

                            resp3.append(generate_results(predictResult, a1,b1,a2,b2,height, width, areaid))
                        else:
                            resp3=["Invalid areacode"]


                    else: # 需要检测
                        
                        # print("FM "+datetime.datetime.now().strftime("%Y%m%d %H%M%S ")+ pidstr + " "+ 'start boxing: ',picid)
                        # sys.stdout.flush()

                        # box_types, detected_boxes=detect(raw_image, detect_net, transform)


                        if areaid>=-1:

                            if detectionEnabled :
                                td1=time.time()
                                # detected_boxes=yoloDetect(originalfile, yoloModel, names,yolodevice,stride)
                                
                                detected_boxes=inference(originalfile,model)
                                
                                td2=time.time()

                                print("FM "+datetime.datetime.now().strftime("%Y%m%d %H%M%S ")+ pidstr + " "+ 'boxes detected: ',birdpic, len(detected_boxes))
                                sys.stdout.flush()

                                boxes=[]
                                for idx in range(len(detected_boxes)):
                                    box = detected_boxes[idx]

                                    ymin, xmin, ymax, xmax = get_square_coordinates(box, height, width)
                                    
                                    boxes.append([box[0],xmin,ymin,xmax,ymax,box[5]])

                                    # print("box[5]:", idx, box[5], box[0])
                                
                            t3=time.time()
                            tp1=t3
                            tp2=t3
                            
                            if len(boxes)==0:
                                resp3=["No boxes"]

                            elif len(boxes)>0 : # mammals

                                needtocheckqq=False

                                # print("    **** ", boxes)

                                for idx in range(len(boxes)): 
                                    
                                    # boxes: [score,x1,y1,x2,y2]
                                    my_squared_image = cubeTheImage(raw_image,boxes[idx][1],boxes[idx][2],boxes[idx][3],boxes[idx][4], width, height)

                                    tp1=time.time()

                                    if boxes[idx][5]=='fish' or boxes[idx][5]=='shrimp':
                                        predictResult = predict(my_squared_image, net, softmax)
                                        # 当第一个"好"类的置信度小于 0.70 时，需要检测 qq
                                        if idx==0 and boxes[idx][0]<0.70:
                                            needtocheckqq=True
                                    else:
                                        # 当有非鱼类时，需要检测 qq
                                        needtocheckqq=True
                                        if boxes[idx][5]=="bird":
                                            predictResult=[(-1,boxes[idx][0])]
                                        elif boxes[idx][5]=="person" or boxes[idx][5]=="mammals":
                                            predictResult=[(-2,boxes[idx][0])]
                                        elif boxes[idx][5]=="reptile":
                                            predictResult=[(-3,boxes[idx][0])]
                                        elif boxes[idx][5]=="amphibia":
                                            predictResult=[(-4,boxes[idx][0])]
                                        else:
                                            predictResult=[(-5,boxes[idx][0])]
                                    tp2=time.time()

                                    resp3.append(generate_results(predictResult, boxes[idx][1],boxes[idx][2],boxes[idx][3],boxes[idx][4],height, width, areaid, boxes[idx][0]))

                            elif len(boxes)>0 and boxes[0][0]>0.8: #not mammals
                                resp3 = generate_nonFishResult(boxes,height, width)
                        else:
                            resp3=["Invalid areacode"]
                        # endof if areaid>=-1
                    #end of detect the box myself

                if serverid.endswith("r"):
                    # 係用户手工框图后重新发送的图片，不需要检查
                    needtocheckqq=False

                if not needtocheckqq:
                    print("FM "+datetime.datetime.now().strftime("%Y%m%d %H%M%S ")+ " *** No need to check img: ", boxes[0][5] if len(boxes)>0 else "(no box)", boxes[0][0] if len(boxes)>0 else "(no box)")
                else:
                    print("FM "+datetime.datetime.now().strftime("%Y%m%d %H%M%S ")+ " *** Need to check img: ", boxes[0][5] if len(boxes)>0 else "(no box)", boxes[0][0] if len(boxes)>0 else "(no box)")
                sys.stdout.flush()

                if not skipqqimagecheck and not imgsource in donotChkSources and needtocheckqq:

                    risky, traceid = checkQQResult(runningproc, originalfile)

                    if risky=="1":
                        resp=["87014"]
                        resp3=["87014"]
                        
                        shutil.move(originalfile, DIRFOR87014+birdpic)
                        
                if len(resp3)==0:
                    resp3=["No birds"]

                elif yolonobox:
                    if resp3[0]["list"][0][0]<30: # 如果 yolo 失败，且直接识别时首鸟置信度小于30，仍然认为没有鸟；
                        resp3=["No boxes"]

                if len(resp3)>1 and not isinstance(resp3[0], str):
                    sys.stdout.flush()
                    resp3 = cleanupResult(resp3)

                if len(resp3)>0 and not isinstance(resp3[0], str):
                    resp3[0]['w']=width
                    resp3[0]['h']=height
                    resp3[0]['REP']=serverid
                    resp3[0]['dnid']=dnid
                    if areaid>=0:
                        resp3[0]['areaen']=area[areaid][0]
                        resp3[0]['areacn']=area[areaid][1]

                resultjson = json.dumps(resp3,separators=(',', ':'),ensure_ascii=False)

                resultfile=open(TMPDIR+"/"+picid+".jpg.result3.json", "wb", buffering=0)
                resultfile.write(resultjson.encode('utf-8'))
                resultfile.close

                targetdir=picid.split('_')[1]
                if not os.path.isdir(DESTDIR+"/"+targetdir):
                    os.mkdir(DESTDIR+"/"+targetdir,0o777)

                if not os.path.isdir(RESULTDIR+"/"+targetdir):
                    os.mkdir(RESULTDIR+"/"+targetdir,0o777)

                try:
                    os.rename(TMPDIR+"/"+picid+".jpg.result3.json", RESULTDIR+"/"+targetdir+"/"+picid+".jpg.result3.json")
                except:
                    pass
                


                if url!="":
                    try:
                        os.rename(originalfile, DESTDIR+"/"+targetdir+"/"+(birdpic.replace(".url",".jpg")))
                    except:
                        t4=time.time()
                        logger.info(loggingTag+pidstr + " "+'{:1.3f}'.format(t2-t1)+" "+'{:1.3f}'.format(t3-t2)+" "+'{:1.3f}'.format(t4-t3)+" "+'{:1.3f}'.format(t4-t1)+" Failed to fetch: " + url)
                        pass    

                t=1
                t4=time.time()
                print(loggingTag+datetime.datetime.now().strftime("%Y%m%d %H%M%S ")+ pidstr + " "+'{:1.3f}'.format(t2-t1)+" "+'{:1.3f}'.format(t3-t2)+" "+'{:1.3f}'.format(t4-t3)+" "+'{:1.3f}'.format(t4-t1)+" "+birdpic+": "+resultjson)
                logger.info(loggingTag+pidstr + " "+'{:1.3f}'.format(t2-t1)+" "+'{:1.3f}'.format(t3-t2)+" "+'{:1.3f}'.format(t4-t3)+" "+'{:1.3f}'.format(t4-t1)+" "+birdpic+": "+resultjson)
                sys.stdout.flush()

            

        #endfor

        sys.stdout.flush()

        if t==0:
            checkQuit()
            mySleep()
            currentday=datetime.datetime.now().strftime("%Y%m%d")
            if currentday!=lastday:
                logger.info("FM " + pidstr + " A new day.")
                lastday=currentday

            if not skipqqimagecheck:
                lasthour, newdyweixintoken = getweixintoken(lasthour)
                if newdyweixintoken!="":
                    dyweixintoken=newdyweixintoken

    #end of while

