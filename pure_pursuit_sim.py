#!/usr/bin/python3

#================================================================
# File name: pure_pursuit_sim.py                                                                  
# Description: pure pursuit controller for GEM vehicle in Gazebo                                                              
# Author: Hang Cui
# Email: hangcui3@illinois.edu                                                                     
# Date created: 07/10/2021                                                                
# Date last modified: 07/15/2021                                                          
# Version: 0.1                                                                    
# Usage: rosrun gem_pure_pursuit_sim pure_pursuit_sim.py                                                                    
# Python version: 3.8      

from skimage import morphology
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from sensor_msgs.msg import CameraInfo

# Define a class to receive the characteristics of each line detection
class Line():
	def __init__(self, n):
		"""
		n is the window size of the moving average
		"""
		self.n = n
		self.detected = False

		# Polynomial coefficients: x = A*y^2 + B*y + C
		# Each of A, B, C is a "list-queue" with max length n
		self.A = []
		self.B = []
		self.C = []
		# Average of above
		self.A_avg = 0.
		self.B_avg = 0.
		self.C_avg = 0.

	def get_fit(self):
		return (self.A_avg, self.B_avg, self.C_avg)

	def add_fit(self, fit_coeffs):
		"""
		Gets most recent line fit coefficients and updates internal smoothed coefficients
		fit_coeffs is a 3-element list of 2nd-order polynomial coefficients
		"""
		# Coefficient queue full?
		q_full = len(self.A) >= self.n

		# Append line fit coefficients
		self.A.append(fit_coeffs[0])
		self.B.append(fit_coeffs[1])
		self.C.append(fit_coeffs[2])

		# Pop from index 0 if full
		if q_full:
			_ = self.A.pop(0)
			_ = self.B.pop(0)
			_ = self.C.pop(0)

		# Simple average of line coefficients
		self.A_avg = np.mean(self.A)
		self.B_avg = np.mean(self.B)
		self.C_avg = np.mean(self.C)

		return (self.A_avg, self.B_avg, self.C_avg)

#================================================================

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# ROS Headers
import rospy
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import Twist, Vector3
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# Gazebo Headers
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState

def perspective_transform(img):
    
    src1 = [9*640//22, 13*480//20]
    src2 = [18*640//25, 13*480//20]
    src3 = [86*640//100, 17*480//20]
    src4 = [30*640//100, 17*480//20]

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


def line_fit(binary_warped):
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	midpoint = int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[100:midpoint]) + 100
	rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint
	nwindows = 9
	window_height = int(binary_warped.shape[0]/nwindows)
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	leftx_current = leftx_base
	rightx_current = rightx_base
	margin = 100
	minpix = 50
	
	left_lane_inds = []
	right_lane_inds = []

	for window in range(nwindows):
		win_y_low = binary_warped.shape[0] - (window + 1) * window_height
		win_y_high = binary_warped.shape[0] - window * window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin

		cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2)
		cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2)

		left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

		left_lane_inds.append(left_inds)
		right_lane_inds.append(right_inds)

		if len(left_inds) > minpix:
			leftx_current = int(np.mean(nonzerox[left_inds]))
		if len(right_inds) > minpix:
			rightx_current = int(np.mean(nonzerox[right_inds]))

	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]


	try:
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)
	except TypeError:
		return None

	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['out_img'] = out_img
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds

	return ret

def bird_fit(binary_warped, ret, save_file=None):

	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	window_img = np.zeros_like(out_img)
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	margin = 100  # NOTE: Keep this in sync with *_fit()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
	cv2.imwrite('exists.jpg',result)

	plt.imshow(result)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)

	return result


def final_viz(undist, left_fit, right_fit, m_inv, image_point=None, draw_image_point=False):
    ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    color_warp = np.zeros((720, 1280, 3), dtype='uint8')  

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
    undist = np.array(undist, dtype=np.uint8)
    newwarp = np.array(newwarp, dtype=np.uint8)
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    if draw_image_point and image_point is not None:
        cv2.circle(result, (int(image_point[0]), int(image_point[1])), 5, (255, 0, 0), -1)
        src1 = [7*640//22, 13*480//20]
        src2 = [16*640//25, 13*480//20]
        src3 = [85*640//100, 17*480//20]
        src4 = [15*640//100, 17*480//20]
        src_list = [src1,src2,src3,src4]
        for src in src_list:
            cv2.circle(result, (src[0], src[1]), 5, (0, 0, 255), -1)


    cv2.imwrite('chigga.png',result)

    return result


class PurePursuit(object):
        
    def __init__(self):
        rospy.init_node('pure_pursuit_lane_follower',anonymous=True)
        self.bridge = CvBridge()
        self.rate       = rospy.Rate(20)

        self.look_ahead = 8    # meters
        self.wheelbase  = 1.75 # meters
        self.goal       = 0

        self.camera_info_sub = rospy.Subscriber("/front_single_camera/camera_info", CameraInfo, self.camera_info_callback, queue_size=1)
        self.image_sub = rospy.Subscriber("/front_single_camera/image_raw", Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)

        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True
        self.idx = 0

        self.dynamic_waypoint = [self.get_gem_pose()[0] + 0.0001, self.get_gem_pose()[1] + 0.0001]
        self.ackermann_msg = AckermannDrive()
        self.ackermann_msg.steering_angle_velocity = 0.0
        self.ackermann_msg.acceleration            = 0.0
        self.ackermann_msg.jerk                    = 0.0
        self.ackermann_msg.speed                   = 0.0 
        self.ackermann_msg.steering_angle          = 0.0

        self.ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=1)
    
    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.K).reshape(3,3)
        self.dist_coefficients = np.array(msg.D)

    def img_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            processed_img, birdseye_img, absolute_future_waypoint = self.detection(cv_image)

            processed_img_msg = self.bridge.cv2_to_imgmsg(processed_img, "bgr8")
            self.pub_image.publish(processed_img_msg)
            birdseye_img_msg = self.bridge.cv2_to_imgmsg(birdseye_img, "bgr8")
            self.pub_bird.publish(birdseye_img_msg)

            distance = self.dist(self.dynamic_waypoint, [self.get_gem_pose()[0],self.get_gem_pose()[1]])
            print("Current goal: ", self.dynamic_waypoint)
            print("Current location: ", self.get_gem_pose()[0],self.get_gem_pose()[1], distance,"m away")
            if distance < 1:
                self.dynamic_waypoint = absolute_future_waypoint
            
        except: 
            print("=")

    def gradient_thresh(self, img, thresh_min=100, thresh_max=255):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        scale_factor = np.max(grad_mag)/255
        grad_mag = (grad_mag/scale_factor).astype(np.uint8)
        _, binary_output = cv2.threshold(grad_mag, thresh_min, thresh_max, cv2.THRESH_BINARY)

        return binary_output


    def color_thresh(self, img, thresh=(100, 255)):

        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        binary_output1 = cv2.inRange(hls, (0, 200, 0), (180, 255, 30))
        binary_output2 = cv2.inRange(hls, (36, 0, 0), (70, 255,255))
        binary_output3 = cv2.inRange(hls, (60, 40, 40), (130,255, 255))
        binary_output = ~binary_output1 ^ binary_output2 ^ binary_output3
        binary_output = binary_output / 255
        binary_output = binary_output.astype(np.uint8)

        return binary_output


    def combinedBinaryImage(self, img):

        SobelOutput = self.gradient_thresh(img)
        binaryImage = np.zeros_like(SobelOutput)
        ColorOutput = self.color_thresh(img)
        binaryImage[(ColorOutput==1)|(SobelOutput==1)] = 1
        binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)
        binaryImage = (binaryImage.astype(np.uint8))

        return binaryImage


    def perspective_transform(self, img, verbose=False):
        src1 = [9*640//22, 13*480//20]
        src2 = [17*640//25, 13*480//20]
        src3 = [87*640//100, 17*480//20]
        src4 = [30*640//100, 17*480//20]
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

    def detection(self, img):
        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)
        ret = line_fit(img_birdeye)
        
        if ret:
            bird_fit_img = bird_fit(img_birdeye, ret)
            final_img = final_viz(img, ret['left_fit'], ret['right_fit'], Minv)

            top_row = 310  

            left_fitx_topmost = ret['left_fit'][0]*top_row**2 + ret['left_fit'][1]*top_row + ret['left_fit'][2]
            right_fitx_topmost = ret['right_fit'][0]*top_row**2 + ret['right_fit'][1]*top_row + ret['right_fit'][2]
  
            left_point = np.array([[left_fitx_topmost,top_row]], dtype=np.float32).reshape(1, 1, 2)
            right_point = np.array([[right_fitx_topmost,top_row]], dtype=np.float32).reshape(1, 1, 2)

            left_fitx = cv2.perspectiveTransform(left_point, Minv)[0][0][0]
            right_fitx = cv2.perspectiveTransform(right_point, Minv)[0][0][0]

            lane_midpoint_x = (left_fitx+right_fitx)/2

            final_img = final_viz(img, ret['left_fit'], ret['right_fit'], Minv, image_point=(lane_midpoint_x, top_row), draw_image_point=True)

            current_x, current_y, current_yaw = self.get_gem_pose()
            
            absolute_future_waypoint = self.calculate_absolute_future_waypoint(lane_midpoint_x, top_row, current_x, current_y, current_yaw, Minv, ret)

            return final_img, bird_fit_img, absolute_future_waypoint


    def calculate_absolute_future_waypoint(self, lane_midpoint_x, top_row, current_x, current_y, current_yaw, Minv, ret):
        
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

    def get_gem_pose(self):

        rospy.wait_for_service('/gazebo/get_model_state')
        
        try:
            service_response = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            model_state = service_response(model_name='gem')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: " + str(exc))

        x = model_state.pose.position.x
        y = model_state.pose.position.y

        orientation_q      = model_state.pose.orientation
        orientation_list   = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

        return round(x,4), round(y,4), round(yaw,4)

    def start_pp(self):
        while not rospy.is_shutdown():
            curr_x, curr_y, curr_yaw = self.get_gem_pose()
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
            angle = np.clip(angle, -0.61, 0.61) 

            speed = 0
          
            
            self.ackermann_msg.speed = speed
            self.ackermann_msg.steering_angle = angle
            self.ackermann_pub.publish(self.ackermann_msg)
            
            self.rate.sleep()

def pure_pursuit():
    pp = PurePursuit()

    try:
        pp.start_pp()

    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    pure_pursuit()

###############################################################################


