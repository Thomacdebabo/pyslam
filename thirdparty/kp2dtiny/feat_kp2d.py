#!/usr/bin/env python
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2018
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Daniel DeTone (ddetone)
#                       Tomasz Malisiewicz (tmalisiewicz)
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%


import argparse
import glob
import numpy as np
import os
import time

import cv2
import torch
from thirdparty.kp2dtiny.tiny_keypoint_net import KeypointNetRaw, KP2D_TINY
# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')


class KP2DtinyFrontend(object):
  """ Wrapper around pytorch net to help with pre and post image processing. """
  def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh,
               cuda=False):
    self.name = 'KP2Dtiny'
    self.cuda = cuda
    self.nms_dist = nms_dist
    self.conf_thresh = conf_thresh
    self.nn_thresh = nn_thresh # L2 descriptor distance for good match.
    self.cell = 8 # Size of each output cell. Keep this fixed.
    self.border_remove = 4 # Remove points this close to the border.
    self.v2_seg = True
    # Load the network in inference mode.
    self.net = KeypointNetRaw(**KP2D_TINY, v2_seg=self.v2_seg, nClasses=28)
    if cuda:
      # Train on GPU, deploy on GPU.
      self.net.load_state_dict(torch.load(weights_path))
      self.net = self.net.cuda()
    else:
      # Train on GPU, deploy on CPU.
      self.net.load_state_dict(torch.load(weights_path,
                               map_location=lambda storage, loc: storage)['state_dict'])
    self.net.eval()
    self.net.training = False

  def run(self, img):
    """ Process a numpy image to extract points and descriptors.
    Input
      img - HxW numpy float32 input image in range [0,1].
    Output
      corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      desc - 256xN numpy array of corresponding unit normalized descriptors.
      heatmap - HxW numpy heatmap in range [0,1] of point confidences.
      """
    assert img.dtype == np.float32, 'Image must be flo5000at32.'
    C, H, W = img.shape
    inp = img.copy()
    inp = torch.from_numpy(inp)
    inp = torch.autograd.Variable(inp).view(1, C, H, W)
    if self.cuda:
      inp = inp.cuda()
    # Forward pass of network.
    with torch.no_grad():
      score, coord, feat, _, seg = self.net.forward(inp)
    score = torch.cat([coord, score], dim=1).view(3, -1).t().cpu().numpy()
    print(seg.shape)
    
    numbers_to_check = [0,21]  # Add the numbers you want to check here
    debug = cv2.resize(seg[0,0].cpu().numpy()/28, (W, H))
    debug_s =  np.isin(seg[0,0].cpu().numpy(), numbers_to_check).astype("float32")
    debug_s = cv2.resize(debug_s, (W, H)) 
    cv2.imshow('seg', debug)
    cv2.imshow('seg_s', debug_s)
    cv2.waitKey(1)
    feat = feat.view(32, -1).t().cpu().numpy()
    
    seg_mask = ~np.isin(seg.view(-1).cpu().numpy(), numbers_to_check)
    
    mask = (score[:, 2] > self.nn_thresh) #& seg_mask
 
    # Filter based on confidence threshold
    feat = feat[mask, :]
    pts = score[mask, :]
    
    return pts.copy(), feat.copy()

