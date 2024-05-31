#!/usr/bin/env python3

#================================================================
# File name: gem_gnss_pp_tracker_pid.py                                                                  
# Description: gnss waypoints tracker using pid and pure pursuit                                                                
# Author: Hang Cui
# Email: hangcui3@illinois.edu                                                                     
# Date created: 08/02/2021                                                                
# Date last modified: 08/15/2022                                                          
# Version: 1.0                                                                   
# Usage: rosrun gem_gnss gem_gnss_pp_tracker.py                                                                      
# Python version: 3.8                                                             
#================================================================

from __future__ import print_function

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal

# ROS Headers
import alvinxy.alvinxy as axy # Import AlvinXY transformation module
import rospy

# GEM Sensor Headers
from std_msgs.msg import String, Bool, Float32, Float64
from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva

# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt

# Banana Code
from skimage import morphology
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from sensor_msgs.msg import CameraInfo
from line_fit import line_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from cv_bridge import CvBridge


class PID(object):

    def __init__(self, kp, ki, kd, wg=None):

        self.iterm  = 0
        self.last_t = None
        self.last_e = 0
        self.kp     = kp
        self.ki     = ki
        self.kd     = kd
        self.wg     = wg
        self.derror = 0

    def reset(self):
        self.iterm  = 0
        self.last_e = 0
        self.last_t = None

    def get_control(self, t, e, fwd=0):

        if self.last_t is None:
            self.last_t = t
            de = 0
        else:
            de = (e - self.last_e) / (t - self.last_t)

        if abs(e - self.last_e) > 0.5:
            de = 0

        self.iterm += e * (t - self.last_t)

        # take care of integral winding-up
        if self.wg is not None:
            if self.iterm > self.wg:
                self.iterm = self.wg
            elif self.iterm < -self.wg:
                self.iterm = -self.wg

        self.last_e = e
        self.last_t = t
        self.derror = de

        return fwd + self.kp * e + self.ki * self.iterm + self.kd * de


class OnlineFilter(object):

    def __init__(self, cutoff, fs, order):
        
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq

        # Get the filter coefficients 
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)

        # Initialize
        self.z = signal.lfilter_zi(self.b, self.a)
    
    def get_data(self, data):
        filted, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
        return filted


class PurePursuit(object):
    
    def __init__(self):

        self.rate       = rospy.Rate(20)

        self.look_ahead = 4
        self.wheelbase  = 1.75 # meters
        self.offset     = 0.46 # meters
    
        self.camera_info_sub = rospy.Subscriber("/zed2/zed_node/rgb/camera_info", CameraInfo, self.camera_info_callback, queue_size=1)
        # self.image_sub = rospy.Subscriber("zed2/zed_node/stereo/image_rect_color", Image, self.img_callback, queue_size=1)
        self.image_sub = rospy.Subscriber("lane_detection/lane_line/zed", Image, self.img_callback, queue_size=1)
        
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        self.bridge = CvBridge()
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True
        self.idx = 0

        self.stop_sign_sub = rospy.Subscriber("stop_sign/score", Float64, self.stop_sign_callback, queue_size=1)
        self.shouldStop = False

        
        self.ackermann_msg = AckermannDrive()
        self.ackermann_msg.steering_angle_velocity = 0.0
        self.ackermann_msg.acceleration            = 0.0
        self.ackermann_msg.jerk                    = 0.0
        self.ackermann_msg.speed                   = 0.0 
        self.ackermann_msg.steering_angle          = 0.0

        

        # self.dynamic_waypoint = [self.get_gem_pose()[0] + 0.0001, self.get_gem_pose()[1] + 0.0001]

        self.gnss_sub   = rospy.Subscriber("/novatel/inspva", Inspva, self.inspva_callback)
        self.lat        = 0.0
        self.lon        = 0.0
        self.heading    = 0.0
        
        self.enable_sub = rospy.Subscriber("/pacmod/as_tx/enable", Bool, self.enable_callback)

        self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
        self.speed      = 2
        self.ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=1)

        self.olat       = 40.0928563
        self.olon       = -88.2359994

        # read waypoints into the system 
        self.goal       = 0            
        self.read_waypoints() 

        self.desired_speed = 2.5 # m/s, reference speed
        self.max_accel     = 0.48 # % of acceleration
        self.pid_speed     = PID(0.5, 0.0, 0.1, wg=20)
        self.speed_filter  = OnlineFilter(1.2, 30, 4)
        self.bridge = CvBridge()
        # -------------------- PACMod setup --------------------

        self.gem_enable    = False
        self.pacmod_enable = False
        self.dynamic_waypoint = [self.get_gem_state()[0] + 0.0001, self.get_gem_state()[1] + 0.0001]
        # GEM vehicle enable, publish once
        self.enable_pub = rospy.Publisher('/pacmod/as_rx/enable', Bool, queue_size=1)
        self.enable_cmd = Bool()
        self.enable_cmd.data = False

        # GEM vehicle gear control, neutral, forward and reverse, publish once
        self.gear_pub = rospy.Publisher('/pacmod/as_rx/shift_cmd', PacmodCmd, queue_size=1)
        self.gear_cmd = PacmodCmd()
        self.gear_cmd.ui16_cmd = 2 # SHIFT_NEUTRAL

        # GEM vehilce brake control
        self.brake_pub = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
        self.brake_cmd = PacmodCmd()
        self.brake_cmd.enable = False
        self.brake_cmd.clear  = True
        self.brake_cmd.ignore = True

        # GEM vechile forward motion control
        self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.accel_cmd = PacmodCmd()
        self.accel_cmd.enable = False
        self.accel_cmd.clear  = True
        self.accel_cmd.ignore = True

        # GEM vechile turn signal control
        self.turn_pub = rospy.Publisher('/pacmod/as_rx/turn_cmd', PacmodCmd, queue_size=1)
        self.turn_cmd = PacmodCmd()
        self.turn_cmd.ui16_cmd = 1 # None

        # GEM vechile steering wheel control
        self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)
        self.steer_cmd = PositionWithSpeed()
        self.steer_cmd.angular_position = 0.0 # radians, -: clockwise, +: counter-clockwise
        self.steer_cmd.angular_velocity_limit = 2.0 # radians/second

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.K).reshape(3,3)
        self.dist_coefficients = np.array(msg.D)

    def inspva_callback(self, inspva_msg):
        self.lat     = inspva_msg.latitude  # latitude
        self.lon     = inspva_msg.longitude # longitude
        self.heading = inspva_msg.azimuth   # heading in degrees

    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 3) # forward velocity in m/s

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def img_callback(self, data):
        try:
    
            cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
            # pub_img = self.bridge.imgmsg_to_cv2(self.camera_info_sub, "bgr8")
            pub_img = cv_image.copy()
            # processed_img, birdseye_img, absolute_future_waypoint = self.detection(cv_image)
            processed_img, img_point, absolute_future_waypoint = self.detection(cv_image)
            # cv2.cvtColor(pub_img, cv2.COLOR_GRAY2RGB)
            # cv2.circle(pub_img, img_point, radius=10, thickness=2, color=(0, 255, 0))
            # pub_img = self.bridge.cv2_to_imgmsg(pub_img, "bgr8")
            # self.pub_image.publish(pub_img)
            # birdseye_img_msg = self.bridge.cv2_to_imgmsg(birdseye_img, "bgr8")
            # self.pub_bird.publish(birdseye_img_msg)

            print("Current goal: ", self.dynamic_waypoint)

            self.dynamic_waypoint = absolute_future_waypoint
            
        except Exception as e: 
            print(e)

    def heading_to_yaw(self, heading_curr):
        if (heading_curr >= 270 and heading_curr < 360):
            yaw_curr = np.radians(450 - heading_curr)
        else:
            yaw_curr = np.radians(90 - heading_curr)
        return yaw_curr

    def front2steer(self, f_angle):
        
        if(f_angle > 35):
            f_angle = 35
        if (f_angle < -35):
            f_angle = -35
        if (f_angle > 0):
            steer_angle = round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        elif (f_angle < 0):
            f_angle = -f_angle
            steer_angle = -round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        else:
            steer_angle = 0.0
        return 0
        return steer_angle

    def read_waypoints(self):
        # read recorded GPS lat, lon, heading
        dirname  = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../waypoints/xyhead_demo_pp.csv')
        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f)]
        # x towards East and y towards North
        self.path_points_lon_x   = [float(point[0]) for point in path_points] # longitude
        self.path_points_lat_y   = [float(point[1]) for point in path_points] # latitude
        self.path_points_heading = [float(point[2]) for point in path_points] # heading
        self.wp_size             = len(self.path_points_lon_x)
        self.dist_arr            = np.zeros(self.wp_size)

    def detection(self, img):
        # binary_img = self.combinedBinaryImage(img)
        
        # img_birdeye, M, Minv = self.perspective_transform(binary_img)

        # ret = line_fit(img_birdeye)
        # print(ret)
      
        # if ret:
        #     bird_fit_img = bird_fit(img_birdeye, ret)

        #     final_img = final_viz(img, ret['left_fit'], ret['right_fit'], Minv)

        middle_row_index = img.shape[0]//2

        middle_row = img[middle_row_index]
        # print(np.unique(img), img.shape, middle_row.shape)
        
        center_index = img.shape[1]//2
        
        left_index = -1

        for i in range(center_index, -1, -1):
            if middle_row[i] == 255:
                left_index = i
                break
        if left_index == -1:
            left_index = 0
        right_index = -1
        for i in range(center_index,img.shape[1]):
            if middle_row[i] == 255:
                right_index = i
                break
        if right_index == -1:
            right_index = img.shape[1]-1
        average_x = (left_index + right_index) //2
        
        img_pnt = (average_x , middle_row_index)

        # final_img = final_viz(img, ret['left_fit'], ret['right_fit'], Minv, image_point=img_pnt, draw_image_point=True)
          
        current_x, current_y, current_yaw = self.get_gem_state()
        
        absolute_future_waypoint = self.calculate_absolute_future_waypoint(average_x, middle_row_index, current_x, current_y, current_yaw)

        return None, img_pnt, absolute_future_waypoint
        
 


    def calculate_absolute_future_waypoint(self, lane_midpoint_x, top_row, current_x, current_y, current_yaw):
        
        normalized_x, normalized_y, _ = np.linalg.inv(self.camera_matrix).dot(np.array([lane_midpoint_x, top_row, 1.0]))

        scale = -1.5 / normalized_y 

        ground_x = scale * normalized_x
        ground_y = scale  

        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)
        
        world_x = current_x - (sin_yaw * ground_x + cos_yaw * ground_y)
        world_y = current_y + (cos_yaw * ground_x - sin_yaw * ground_y)

        return [world_x, world_y]


    # computes the Euclidean distance between two 2D points
    def dist(self, p1, p2):
        return round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)

    # find the angle bewtween two vectors    
    def find_angle(self, v1, v2):
        cosang = np.dot(v1, v2)
        sinang = la.norm(np.cross(v1, v2))
        # [-pi, pi]
        return np.arctan2(sinang, cosang)

    def stop_sign_callback(self, msg):
        # print("stop sign callback being called", msg.data)
        if msg.data > 0.035:
            print("stopping now", msg.data)
            self.desired_speed = 0
            # self.shouldStop = True
        else:
            self.desired_speed = 1.2
            # self.shouldStop = False

    def wps_to_local_xy(self, lon_wp, lat_wp):
        # convert GNSS waypoints into local fixed frame reprented in x and y
        lon_wp_x, lat_wp_y = axy.ll2xy(lat_wp, lon_wp, self.lat, self.lon)
        return lon_wp_x, lat_wp_y   

    def get_gem_state(self):

        # vehicle gnss heading (yaw) in degrees
        # vehicle x, y position in fixed local frame, in meters
        # reference point is located at the center of GNSS antennas
        # print("long and lat are", self.lon, self.lat)
        local_x_curr, local_y_curr = self.wps_to_local_xy(self.olon, self.olat)

        # heading to yaw (degrees to radians)
        # heading is calculated from two GNSS antennas
        curr_yaw = self.heading_to_yaw(self.heading) 

        # reference point is located at the center of rear axle
        curr_x = local_x_curr - self.offset * np.cos(curr_yaw)
        curr_y = local_y_curr - self.offset * np.sin(curr_yaw)

        return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)
    

    def gradient_thresh(self, img, thresh_min=20, thresh_max=255):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        #1. Convert the image to gray scale
        #2. Gaussian blur the image
        #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        #4. Use cv2.addWeighted() to combine the results
        #5. Convert each pixel to uint8, then apply threshold to get binary image

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
        return binary_image


    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        #1. Apply sobel filter and color filter on input image
        #2. Combine the outputs
        ## Here you can use as many methods as you want.

        ## TODO
        SobelOutput = self.gradient_thresh(img)
        return SobelOutput

    def perspective_transform(self, img, verbose=False):
        src1 = [4*len(img[0])//10, 5*len(img)//10]
        src2 = [6*len(img[0])//10, 5*len(img)//10]
        src3 = [9*len(img[0])//10, 9*len(img)//10]
        src4 = [len(img[0])//10, 9*len(img)//10]

        dest1 = [0, 0]
        dest2 = [len(img[0]), 0]
        dest3 = [len(img[0]), len(img)]
        dest4 = [0, len(img)]

        srcs = np.float32([src1, src2, src3, src4])
        dests = np.float32([dest1, dest2, dest3, dest4])

        M = cv2.getPerspectiveTransform(srcs, dests)
        warped_img = cv2.warpPerspective(img, M, (len(img[0]), len(img)))
        Minv = np.linalg.inv(M)

        return warped_img, M, Minv
        # pt_top_left = [475, 450]
        # pt_bottom_left = [350, 645]
        # pt_bottom_right = [940, 645]
        # pt_top_right = [760, 450]
    
        # input_pts = np.float32([pt_top_left, pt_bottom_left, pt_bottom_right, pt_top_right])
        # width_top = np.sqrt(((pt_top_left[0] - pt_top_right[0]) ** 2) + ((pt_top_left[1] - pt_top_right[1]) ** 2))
        # width_bottom = np.sqrt(((pt_bottom_left[0] - pt_bottom_right[0]) ** 2) + ((pt_bottom_left[1] - pt_bottom_right[1]) ** 2))
        # maxWidth = max(int(width_top), int(width_bottom))
        
        # height_left = np.sqrt(((pt_top_left[0] - pt_bottom_left[0]) ** 2) + ((pt_top_left[1] - pt_bottom_left[1]) ** 2))
        # height_right = np.sqrt(((pt_bottom_right[0] - pt_top_right[0]) ** 2) + ((pt_bottom_right[1] - pt_top_right[1]) ** 2))
        # maxHeight = max(int(height_left), int(height_right))
        # output_pts = np.float32([[ 130, 0 ],
        #                         [ 130, 500 ],
        #                         [ 500, 500 ],
        #                         [ 500, 30   ]]) 
        # M = cv2.getPerspectiveTransform(input_pts,output_pts)

        # Minv = cv2.getPerspectiveTransform(output_pts,input_pts)

        # up_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # warped_img = cv2.warpPerspective(up_img, M, (600, 500), flags=cv2.INTER_LINEAR)
        # warped_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)

        # return warped_img, M, Minv

    def find_angle(self, v1, v2):
        cosang = np.dot(v1, v2)
        sinang = la.norm(np.cross(v1, v2))
        # [-pi, pi]
        return np.arctan2(sinang, cosang)

    # computes the Euclidean distance between two 2D points
    def dist(self, p1, p2):
        return round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)

    def start_pp(self):
        while not rospy.is_shutdown():

            if (self.gem_enable == False):

                        if(self.pacmod_enable == True):

                            # ---------- enable PACMod ----------

                            # enable forward gear
                            self.gear_cmd.ui16_cmd = 3

                            # enable brake
                            self.brake_cmd.enable  = True
                            self.brake_cmd.clear   = False
                            self.brake_cmd.ignore  = False
                            self.brake_cmd.f64_cmd = 0.0

                            # enable gas 
                            self.accel_cmd.enable  = True
                            self.accel_cmd.clear   = False
                            self.accel_cmd.ignore  = False
                            self.accel_cmd.f64_cmd = 0.0

                            self.gear_pub.publish(self.gear_cmd)
                            print("Foward Engaged!")

                            self.turn_pub.publish(self.turn_cmd)
                            print("Turn Signal Ready!")
                            
                            self.brake_pub.publish(self.brake_cmd)
                            print("Brake Engaged!")

                            self.accel_pub.publish(self.accel_cmd)
                            print("Gas Engaged!")

                            self.gem_enable = True

            curr_x, curr_y, curr_yaw = self.get_gem_state()
            # print(curr_x, curr_y, curr_yaw)
            dynamic_goal_x, dynamic_goal_y = self.dynamic_waypoint

            gvcx = dynamic_goal_x - curr_x
            gvcy = dynamic_goal_y - curr_y
            goal_x_veh_coord = gvcx * np.cos(curr_yaw) + gvcy * np.sin(curr_yaw)
            goal_y_veh_coord = gvcy * np.cos(curr_yaw) - gvcx * np.sin(curr_yaw)

            L = np.sqrt(goal_x_veh_coord ** 2 + goal_y_veh_coord ** 2)
            
            alpha = np.arctan2(goal_y_veh_coord , goal_x_veh_coord )
            k = 0.285  
            angle_i = np.arctan((2 * self.wheelbase * np.sin(alpha)) / L)
            angle = angle_i * 2

            f_delta = round(np.clip(angle, -0.61, 0.61), 3)

            f_delta_deg = np.degrees(f_delta)

            # steering_angle in degrees
            steering_angle = self.front2steer(f_delta_deg)

            # if self.shouldStop:
            #     # output_accel = -1
            #     self.desired_speed = 0
            #     print("shouldStop flag is enabled")
            current_time = rospy.get_time()
            filt_vel     = self.speed_filter.get_data(self.speed)
            output_accel = self.pid_speed.get_control(current_time, self.desired_speed - filt_vel)

            if output_accel > self.max_accel:
                output_accel = self.max_accel

            if output_accel < 0.3:
                output_accel = 0.3


            if (f_delta_deg <= 30 and f_delta_deg >= -30):
                self.turn_cmd.ui16_cmd = 1
            elif(f_delta_deg > 30):
                self.turn_cmd.ui16_cmd = 2 # turn left
            else:
                self.turn_cmd.ui16_cmd = 0 # turn right

            self.accel_cmd.f64_cmd = output_accel
            self.steer_cmd.angular_position = np.radians(steering_angle)
            self.accel_pub.publish(self.accel_cmd)
            self.steer_pub.publish(self.steer_cmd)
            self.turn_pub.publish(self.turn_cmd)

            
            self.rate.sleep()


def pure_pursuit():

    rospy.init_node('gnss_pp_node', anonymous=True)
    pp = PurePursuit()

    try:
        pp.start_pp()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    pure_pursuit()


