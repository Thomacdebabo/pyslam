"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import cv2
import math

from config import Config

from visual_odometry import VisualOdometry
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset import dataset_factory

#from mplot3d import Mplot3d
#from mplot2d import Mplot2d
from mplot_thread import Mplot2d, Mplot3d

from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_manager import feature_manager_factory
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from feature_matcher import feature_matcher_factory, FeatureMatcherTypes

from feature_tracker_configs import FeatureTrackerConfigs
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Visual Odometry')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='config file')
    parser.add_argument('--dataset', type=str, default='dataset/kitti', help='dataset folder')
    parser.add_argument('--output', type=str, default='output', help='output folder')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--use_viewer', action='store_true', help='use pangolin viewer')
    parser.add_argument('--plot', action='store_true', help='plot results')
    parser.add_argument('--tracker', type=str, default='LK_SHI_TOMASI', help='feature tracker')
    args = parser.parse_args()
    return args

def calculate_statistics(errors, inliers_report, matches_report, fname='stats.json'):
        # Show error statistics
    errors = np.array(errors)
    inliers_report = np.array(inliers_report)
    matches_report = np.array(matches_report)
    ratio = inliers_report/matches_report
    
    print("------------------")
    print('Error statistics:')
    print("------------------")
    print('mean error: ', np.mean(errors))
    print('median error: ', np.median(errors))
    print('max error: ', np.max(errors))
    print('min error: ', np.min(errors))
    print('std error: ', np.std(errors))
    print('RMSE: ', math.sqrt(np.mean(errors**2)))
    print("------------------")
    
    print("------------------")
    print('Matches statistics:')
    print("------------------")
    print('mean matches: ', np.mean(matches_report))
    print('median matches: ', np.median(matches_report))
    print('max matches: ', np.max(matches_report))
    print('min matches: ', np.min(matches_report))
    print('std matches: ', np.std(matches_report))
    print("------------------")
    
    print("------------------")
    print('Inliers statistics:')
    print("------------------")
    print('mean inliers: ', np.mean(inliers_report))
    print('median inliers: ', np.median(inliers_report))
    print('max inliers: ', np.max(inliers_report))
    print('min inliers: ', np.min(inliers_report))
    print('std inliers: ', np.std(inliers_report))
    print("------------------")
    
    print("------------------")
    print('Inliers/matches statistics:')
    print("------------------")
    print('mean ratio: ', np.mean(ratio))
    print('median ratio: ', np.median(ratio))
    print('max ratio: ', np.max(ratio))
    print('min ratio: ', np.min(ratio))
    print('std ratio: ', np.std(ratio))
    print("------------------")
    
    # store statistics as a json file
    import json
    stats = {}
    stats['mean_error'] = np.mean(errors)
    stats['median_error'] = np.median(errors)
    stats['max_error'] = np.max(errors)
    stats['min_error'] = np.min(errors)
    stats['std_error'] = np.std(errors)
    stats['RMSE'] = math.sqrt(np.mean(errors**2))
    stats['mean_matches'] = np.mean(matches_report)
    stats['median_matches'] = np.median(matches_report)
    stats['max_matches'] = np.max(matches_report)
    stats['min_matches'] = np.min(matches_report)
    stats['std_matches'] = np.std(matches_report)
    stats['mean_inliers'] = np.mean(inliers_report)
    stats['median_inliers'] = np.median(inliers_report)
    stats['max_inliers'] = np.max(inliers_report)
    stats['min_inliers'] = np.min(inliers_report)
    stats['std_inliers'] = np.std(inliers_report)
    stats['mean_ratio'] = np.mean(ratio)
    stats['median_ratio'] = np.median(ratio)
    stats['max_ratio'] = np.max(ratio)
    stats['min_ratio'] = np.min(ratio)
    stats['std_ratio'] = np.std(ratio)
    
    # convert all values in stats to python int from numpy
    
    for key in stats.keys():
        stats[key] = int(stats[key])
    with open(fname, 'w') as outfile:
        json.dump(stats, outfile)


"""
use or not pangolin (if you want to use it then you need to install it by using the script install_thirdparty.sh)
"""

kUsePangolin = False




if __name__ == "__main__":

    args = parse_args()
    kUsePangolin = args.use_viewer
    
    
    is_draw_3d = args.plot
    is_draw_traj_img = args.plot
    is_draw_err = args.plot 
    is_draw_matched_points = args.plot 
    is_draw_cam = args.plot
    
    if kUsePangolin:
        from viewer3D import Viewer3D
    config = Config()
    
    config.dataset_settings['is_color'] = "True"

    dataset = dataset_factory(config.dataset_settings)

    groundtruth = groundtruth_factory(config.dataset_settings)

    cam = PinholeCamera(config.cam_settings['Camera.width'], config.cam_settings['Camera.height'],
                        config.cam_settings['Camera.fx'], config.cam_settings['Camera.fy'],
                        config.cam_settings['Camera.cx'], config.cam_settings['Camera.cy'],
                        config.DistCoef, config.cam_settings['Camera.fps'])


    num_features=2000  # how many features do you want to detect and track?

    # select your tracker configuration (see the file feature_tracker_configs.py) 
    # LK_SHI_TOMASI, LK_FAST
    # SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, FAST_TFEAT
    tracker_config = tracker_config = getattr(FeatureTrackerConfigs, args.tracker)
    tracker_config['num_features'] = num_features
    
    feature_tracker = feature_tracker_factory(**tracker_config)

    # create visual odometry object 
    vo = VisualOdometry(cam, groundtruth, feature_tracker)

    
    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    half_traj_img_size = int(0.5*traj_img_size)
    draw_scale = 1

    
    if args.plot:
        if kUsePangolin:
            viewer3D = Viewer3D()
        else:
            plt3d = Mplot3d(title='3D trajectory')

    
        err_plt = Mplot2d(xlabel='img id', ylabel='m',title='error')

        
        matched_points_plt = Mplot2d(xlabel='img id', ylabel='# matches',title='# matches')

    img_id = 0
    errors = []
    matches_report = []
    inliers_report = []
    while dataset.isOk():

        img = dataset.getImage(img_id)

        if img is not None:

            vo.track(img, img_id)  # main VO function 

            if(img_id > 2):	       # start drawing from the third image (when everything is initialized and flows in a normal way)

                x, y, z = vo.traj3d_est[-1]
                x_true, y_true, z_true = vo.traj3d_gt[-1]

                if is_draw_traj_img:      # draw 2D trajectory (on the plane xz)
                    draw_x, draw_y = int(draw_scale*x) + half_traj_img_size, half_traj_img_size - int(draw_scale*z)
                    true_x, true_y = int(draw_scale*x_true) + half_traj_img_size, half_traj_img_size - int(draw_scale*z_true)
                    cv2.circle(traj_img, (draw_x, draw_y), 1,(img_id*255/4540, 255-img_id*255/4540, 0), 1)   # estimated from green to blue
                    cv2.circle(traj_img, (true_x, true_y), 1,(0, 0, 255), 1)  # groundtruth in red
                    # write text on traj_img
                    cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
                    text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
                    cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                    # show 		
                    cv2.imshow('Trajectory', traj_img)

                if is_draw_3d:           # draw 3d trajectory 
                    if kUsePangolin:
                        viewer3D.draw_vo(vo)   
                    else:
                        plt3d.drawTraj(vo.traj3d_gt,'ground truth',color='r',marker='.')
                        plt3d.drawTraj(vo.traj3d_est,'estimated',color='g',marker='.')
                        plt3d.refresh()
                err = math.sqrt((x_true-x)**2 + (y_true-y)**2 + (z_true-z)**2)
                errors.append(err)
                if is_draw_err:         # draw error signals 
                    errx = [img_id, math.fabs(x_true-x)]
                    erry = [img_id, math.fabs(y_true-y)]
                    errz = [img_id, math.fabs(z_true-z)] 
                    
                    
                    sqr_err  = [img_id, err]
                    err_plt.draw(errx,'err_x',color='g')
                    err_plt.draw(erry,'err_y',color='b')
                    err_plt.draw(errz,'err_z',color='r')
                    err_plt.draw(sqr_err,'sqr_err',color='k')
                    err_plt.refresh()    
                    

                if is_draw_matched_points:
                    matched_kps_signal = [img_id, vo.num_matched_kps]
                    inliers_signal = [img_id, vo.num_inliers]                    
                    matched_points_plt.draw(matched_kps_signal,'# matches',color='b')
                    matched_points_plt.draw(inliers_signal,'# inliers',color='g')                    
                    matched_points_plt.refresh()
                matches_report.append(vo.num_matched_kps)
                inliers_report.append(vo.num_inliers)                    


            # draw camera image 
            if is_draw_cam:
                cv2.imshow('Camera', vo.draw_img)				

        # press 'q' to exit!
        if is_draw_cam:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        img_id += 1

    #print('press a key in order to exit...')
    #cv2.waitKey(0)
    
    calculate_statistics(errors, inliers_report, matches_report, fname='stats.json')
        
    
    if is_draw_traj_img:
        print('saving map.png')
        cv2.imwrite('map.png', traj_img)
    if is_draw_3d:
        if not kUsePangolin:
            plt3d.quit()
        else: 
            viewer3D.quit()
    if is_draw_err:
        err_plt.quit()
    if is_draw_matched_points is not None:
        matched_points_plt.quit()
                
    cv2.destroyAllWindows()
