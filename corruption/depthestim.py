#!/usr/bin/env python
# ref: https://github.com/sniklaus/3d-ken-burns
import base64
import cupy
import cv2
import flask
import getopt
import gevent
import gevent.pywsgi
import glob
import h5py
import io
import math
import moviepy
import moviepy.editor
import numpy
import os
import random
import re
import scipy
import scipy.io
import shutil
import sys
import tempfile
import time
import torch
import torchvision
import urllib
import zipfile

##########################################################

# requires at least pytorch version 1.2.0
assert (int(str('').join(torch.__version__.split('.')[0:2])) >= 12)

# make sure to not compute gradients for computational performance
torch.set_grad_enabled(False)

# make sure to use cudnn for computational performance
torch.backends.cudnn.enabled = True

##########################################################

objCommon = {}

exec(open('./common.py', 'r').read())

exec(open('./models/disparity-estimation.py', 'r').read())
exec(open('./models/disparity-adjustment.py', 'r').read())
exec(open('./models/disparity-refinement.py', 'r').read())
exec(open('./models/pointcloud-inpainting.py', 'r').read())

##########################################################

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--rgb' and strArgument != '':
        arguments_strIn = strArgument  # path to the input image
    if strOption == '--depth' and strArgument != '':
        arguments_strOut = strArgument  # path to where the output should be stored
# end

##########################################################

if __name__ == '__main__':
    imgs_in = [img for img in os.listdir(
        arguments_strIn) if img.endswith("png")]
    if not (os.path.isdir(arguments_strOut)):
        os.makedirs(arguments_strOut)
    for img in imgs_in:
        print(os.path.join(arguments_strIn, img))
        npyImage = cv2.imread(filename=os.path.join(
            arguments_strIn, img), flags=cv2.IMREAD_COLOR)

        fltFocal = max(npyImage.shape[1], npyImage.shape[0]) / 2.0
        fltBaseline = 40.0

        tenImage = torch.FloatTensor(numpy.ascontiguousarray(npyImage.transpose(
            2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
        tenDisparity = disparity_estimation(tenImage)
        tenDisparity = disparity_refinement(torch.nn.functional.interpolate(input=tenImage, size=(
            tenDisparity.shape[2] * 4, tenDisparity.shape[3] * 4), mode='bilinear', align_corners=False), tenDisparity)
        tenDisparity = torch.nn.functional.interpolate(input=tenDisparity, size=(
            tenImage.shape[2], tenImage.shape[3]), mode='bilinear', align_corners=False) * (max(tenImage.shape[2], tenImage.shape[3]) / 256.0)
        tenDepth = (fltFocal * fltBaseline) / (tenDisparity + 0.0000001)

        npyDisparity = tenDisparity[0, 0, :, :].cpu().numpy()
        npyDepth = tenDepth[0, 0, :, :].cpu().numpy()

        # cv2.imwrite(filename=arguments_strOut.replace('.npy', '.png'), img=(npyDisparity / fltBaseline * 255.0).clip(0.0, 255.0).astype(numpy.uint8))

        numpy.save(arguments_strOut+"/"+img.replace(".png", ".npy"), npyDepth)
# end
