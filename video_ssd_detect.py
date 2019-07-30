
import numpy as np
import matplotlib.pyplot as plt
import cv2
#%matplotlib inline
#from __future__ import * 
#plt.rcParams['figure.figsize'] = (10, 10)
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '/home/huanglong/caffe-ssd'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()


from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = '/home/huanglong/caffe-ssd/VOCdevkit/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

model_def = 'models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
model_weights = 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_70000.caffemodel'
#model_weights = '/home/huanglong/caffe-ssd/output_pruned.caffemodel'
#model_def='/home/huanglong/caffe-ssd/deploy_prune_new.prototxt'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


# set net to batch size of 1
image_resize = 300
net.blobs['data'].reshape(1,3,image_resize,image_resize)



#cap = cv2.VideoCapture("Minions_banana.mp4")
import math 
import os, sys
import glob
from PIL import Image
 
input_dir = '/home/huanglong/pic_test/Traffic_Detection_Test/IMG_0543.MOV'
 
#input_lists = glob.glob(input_dir + '/*.MOV')
#for i in input_lists
#    
#    name=i.split('/')[-1]
#    name=name.split('.')[0]
#    #print name
cap = cv2.VideoCapture(input_dir)
outputFile='/home/huanglong/pic_test/2/IMG_0543.avi'
vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M','J','P','G'), 28, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, image1 = cap.read()
    if ret == True:

       
        #cv2.imshow('img',image)
    #image = caffe.io.load_image('/home/huanglong/002795.jpg')
    #plt.imshow(image)
        #image1=cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        
        img = image1/ 255.0
        transformed_image = transformer.preprocess('data', img)
        #print transformed_image.shape
        net.blobs['data'].data[...] = transformed_image
    
    # Forward pass.
        detections = net.forward()['detection_out']
    # Parse the outputs.()['detection_out']
    
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]
    
    # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
    
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
    
    
    #colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        colors=[(0,0,0),(128,0,0),(0,128,0),(128,128,0)]
    #fig=plt.figure()
    #currentAxis = plt.gca()
    
    
        for i in xrange(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * image1.shape[1]))
            ymin = int(round(top_ymin[i] * image1.shape[0]))
            xmax = int(round(top_xmax[i] * image1.shape[1]))
            ymax = int(round(top_ymax[i] * image1.shape[0]))
            #print xmin, ymin, xmax, ymax
       
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            display_txt = '%s: %.2f'%(label_name, score)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            #print coords,
            color = colors[label]
            cv2.rectangle(image1, (xmin,ymin), (xmax,ymax), color, 4)
        #currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        #currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5}) 
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(image1, display_txt, (xmin,ymin), font, 2, color, 2)
        vid_writer.write(image1)
   # if cv2.waitKey(1000) & 0xFF == ord('q'):
       #     break        
      
    #print "finsh"       

    #
cap.release()
cv2.destroyAllWindows()