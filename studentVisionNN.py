import time
import math
import numpy as np
import cv2
import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology

import torch
import torchvision
from torchvision import transforms

def crop_center(image, crop_height, crop_width):
    """
    Crop the input numpy image to the center with specified height and width.

    Args:
    - image (numpy.ndarray): Input image.
    - crop_height (int): Height of the cropped image.
    - crop_width (int): Width of the cropped image.

    Returns:
    - cropped_image (numpy.ndarray): Cropped image.
    """
    height, width = image.shape[:2]
    start_h = max(0, (height - crop_height) // 2)
    start_w = max(0, (width - crop_width) // 2)
    end_h = min(height, start_h + crop_height)
    end_w = min(width, start_w + crop_width)
    cropped_image = image[start_h:end_h, start_w:end_w]
    return cropped_image

# https://pytorch.org/hub/hustvl_yolop/
# https://github.com/hustvl/YOLOP

class lanenet_detector:
    def __init__(self):

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        # self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        # self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)
        self.sub_image = rospy.Subscriber('zed2/zed_node/rgb/image_rect_color', Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True
        self.idx = 0

        self.model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)


    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        mask_image, bird_image = self.detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)

    def detection(self, img):
        img = crop_center(img, 640, 640)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        if self.model:
            det_out, da_seg_out, ll_seg_out = self.model(img)
            # print(type(det_out)) # , det_out.shape)
            # print(type(da_seg_out)) #, da_seg_out.shape)
            # print(type(ll_seg_out)) #, ll_seg_out.shape)
            
            image_pil = transforms.ToPILImage()(da_seg_out[0])
            image_pil.save("da_seg_out.png")
            image_pil = transforms.ToPILImage()(ll_seg_out[0])
            image_pil.save("ll_seg_out.png")

        image_pil = transforms.ToPILImage()(img[0])
        image_pil.save("plain_img.png")

        return None, None


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)
