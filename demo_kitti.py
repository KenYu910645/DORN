import cv2
import caffe
import numpy as np
import scipy.io as sio
import argparse
import os
import pdb
import glob 

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--filename', type=str, default='./data/KITTI/demo_01.png', help='path to an image')
parser.add_argument('--input_dir', type=str, default='/home/lab530/KenYu/kitti/training/image_2/')
parser.add_argument('--outputroot', type=str, default='/home/lab530/KenYu/kitti/training/image_depth', help='output path')

# caffe.set_mode_gpu()
caffe.set_mode_cpu()
# caffe.set_device(0)
net = caffe.Net('models/KITTI/deploy.prototxt', 'models/KITTI/cvpr_kitti.caffemodel', caffe.TEST)
pixel_means = np.array([[[103.0626, 115.9029, 123.1516]]])

def depth_prediction(filename):
    img = cv2.imread(filename, 1)
    img = img.astype(np.float32)
    H = img.shape[0]
    W = img.shape[1]
    img -= pixel_means
    img = cv2.resize(img, (W, 385), interpolation=cv2.INTER_LINEAR)
    ord_score = np.zeros((385, W), dtype=np.float32)
    counts = np.zeros((385, W), dtype=np.float32)
    for i in xrange(4):
        h0 = 0
        h1 = 385
        w0 = int(0 + i*256)
        w1 = w0 + 513
        if w1 > W:
           w0 = W - 513
           w1 = W

        data = img[h0:h1, w0:w1, :]
        data = data[None, :]
        data = data.transpose(0,3,1,2)
        blobs = {}
        blobs['data'] = data
        net.blobs['data'].reshape(*(blobs['data'].shape))
        forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
        net.forward(**forward_kwargs)
        pred = net.blobs['decode_ord'].data.copy()
        pred = pred[0,0,:,:]
        ord_score[h0:h1,w0:w1] = ord_score[h0:h1, w0:w1] + pred
        counts[h0:h1,w0:w1] = counts[h0:h1, w0:w1] + 1.0

    ord_score = ord_score/counts - 1.0
    ord_score = (ord_score + 40.0)/25.0
    ord_score = np.exp(ord_score)
    ord_score = cv2.resize(ord_score, (W, H), interpolation=cv2.INTER_LINEAR)
    return ord_score
    #ord_score = ord_score*256.0

args = parser.parse_args()


f_list = []
TRAIN_TXT = "/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/chen_split/val.txt"
with open(TRAIN_TXT, "r") as f:
    f_list = f.readlines()
f_list = [f.rstrip("\n") for f in f_list]
print("Total " + str(len(f_list)) + " assigned file found in " + TRAIN_TXT)

# 
exist_fn = [fn.split('.')[0] for fn in os.listdir(args.outputroot)]
print("Total " + str(len(exist_fn)) + " existed file found in " + str(args.outputroot) + ", will try to skip those")

# Filter files if they already exist in target directory
doable_fns = [fn for fn in f_list if not fn in exist_fn]
print("Number of doable files : " + str(len(doable_fns)))

# file_list = glob.glob( os.path.join( args.input_dir, "*.png") )
file_list = [ os.path.join( args.input_dir, fn + ".png")  for fn in doable_fns]
print("Total number of input file " + str(len(file_list)))
# print(file_list)

for i, f_name in enumerate(file_list):
    # depth = depth_prediction(args.filename)
    depth = depth_prediction(f_name)
    depth = depth*256.0
    depth = depth.astype(np.uint16)
    # img_id = args.filename.split('/')
    img_id = f_name.split('/')
    img_id = img_id[len(img_id)-1]
    img_id = img_id[0:len(img_id)-4]
    if not os.path.exists(args.outputroot):
        os.makedirs(args.outputroot)
    cv2.imwrite(str(args.outputroot + '/' + img_id + '.png'), depth)
    
    # print("Process file (" + {str(i)} + "/" + str(len(file_list)) + ")" )
    print("Process file ({}/{})".format(i, len(file_list)))
