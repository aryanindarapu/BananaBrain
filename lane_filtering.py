import numpy as np
import cv2
import torch.backends
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from cv_bridge import CvBridge, CvBridgeError

from utils.utils import \
    time_synchronized,select_device, increment_path,\
    scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result, save_masks, \
    AverageMeter,\
    LoadImages

import torch
import torchvision
from torchvision import transforms
import time
from pathlib import Path

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
    def __init__(self, model, device, save_imgs=False):

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        # self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        # self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)
        # TODO: change the topic names for left and right cameras
        self.zed_sub_img = rospy.Subscriber('zed2/zed_node/rgb/image_rect_color', Image, self.zed_callback, queue_size=1)
        self.right_sub_img = rospy.Subscriber('camera_fr/arena_camera_node/image_raw', Image, self.right_callback, queue_size=1)
        self.left_sub_img = rospy.Subscriber('camera_fl/arena_camera_node/image_raw', Image, self.left_callback, queue_size=1)
        
        # self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        # self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        self.da_zed_pub = rospy.Publisher("lane_detection/driving_area/zed", Image, queue_size=1)
        self.ll_zed_pub = rospy.Publisher("lane_detection/lane_line/zed", Image, queue_size=1)
        self.da_right_pub = rospy.Publisher("lane_detection/driving_area/right", Image, queue_size=1)
        self.ll_right_pub = rospy.Publisher("lane_detection/lane_line/right", Image, queue_size=1)
        self.da_left_pub = rospy.Publisher("lane_detection/driving_area/left", Image, queue_size=1)
        self.ll_left_pub = rospy.Publisher("lane_detection/lane_line/left", Image, queue_size=1)
        
        self.da_zed, self.ll_zed = None, None
        self.da_right, self.ll_right = None, None
        self.da_left, self.ll_left = None, None
        
        # Topic will be subscribed by the lane controller node
        # This topic will be used to determine which camera is currently being used
        self.curr_camera_pub = rospy.Publisher("lane_detection/current_camera", String, queue_size=1)
        
        # NOTE: this currently runs every second, want to make duration smaller
        # self.timer = rospy.Timer(rospy.Duration(1), self.process_cameras)
        
        self.save_imgs = save_imgs

        self.model = model
        self.model.to(device)
        self.device = device
        
    # def process_cameras(self, msg):
    #     pass

    def zed_callback(self, data):
        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        da_mask, ll_mask = self.lane_detection(raw_img)
        # print("zed1", da_mask.shape, ll_mask.shape, type(ll_mask))
        # cv2.imwrite("zed.png", ll_mask)

        # ll_mask = cv2.cvtColor(ll_mask, cv2.COLOR_GRAY2BGR)
        # print("zed2", da_mask.shape, ll_mask.shape, type(ll_mask))
        self.da_zed, self.ll_zed = da_mask, ll_mask
        
        ll_mask = np.array(ll_mask, dtype=np.uint8)
        ll_mask[ll_mask == 1] = 255
        da_mask = np.array(da_mask, dtype=np.uint8)
        da_mask[da_mask == 1] = 255

        if da_mask is not None and ll_mask is not None:          
            # Convert an OpenCV image into a ROS image message
            out_da_msg = self.bridge.cv2_to_imgmsg(da_mask, 'mono8')
            out_ll_msg = self.bridge.cv2_to_imgmsg(ll_mask, 'mono8')

            # Publish image message in ROS
            self.da_zed_pub.publish(out_da_msg)
            self.ll_zed_pub.publish(out_ll_msg)
            
    def right_callback(self, data):
        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        da_mask, ll_mask = self.lane_detection(raw_img)

        self.da_right, self.ll_right = da_mask, ll_mask
        
        ll_mask = np.array(ll_mask, dtype=np.uint8)
        ll_mask[ll_mask == 1] = 255
        da_mask = np.array(da_mask, dtype=np.uint8)
        da_mask[da_mask == 1] = 255

        if da_mask is not None and ll_mask is not None:          
            # Convert an OpenCV image into a ROS image message
            out_da_msg = self.bridge.cv2_to_imgmsg(da_mask, 'mono8')
            out_ll_msg = self.bridge.cv2_to_imgmsg(ll_mask, 'mono8')

            # Publish image message in ROS
            self.da_right_pub.publish(out_da_msg)
            self.ll_right_pub.publish(out_ll_msg)
    
    def left_callback(self, data):
        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        da_mask, ll_mask = self.lane_detection(raw_img)
        self.da_left, self.ll_left = da_mask, ll_mask
        
        ll_mask = np.array(ll_mask, dtype=np.uint8)
        ll_mask[ll_mask == 1] = 255
        da_mask = np.array(da_mask, dtype=np.uint8)
        da_mask[da_mask == 1] = 255

        if da_mask is not None and ll_mask is not None:          
            # Convert an OpenCV image into a ROS image message
            out_da_msg = self.bridge.cv2_to_imgmsg(da_mask, 'mono8')
            out_ll_msg = self.bridge.cv2_to_imgmsg(ll_mask, 'mono8')

            # Publish image message in ROS
            self.da_left_pub.publish(out_da_msg)
            self.ll_left_pub.publish(out_ll_msg)
            
    def lane_detection(self, img):
        try:
            imgsz = 640

            # inf_time = AverageMeter()
            # waste_time = AverageMeter()
            # nms_time = AverageMeter()
            model.eval()

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            
            # print("eache")
            t0 = time.time()
            # for path, img, im0s, vid_cap in dataset: # single image inference
            img = crop_center(img, 704, 1280)
            # cv2.imwrite("./tmp.png", np.array(img))
            # np.array(img.cpu())

            # dataset = LoadImages('./tmp.png', img_size=imgsz, stride=32)
            dataset = LoadImages('./tmp.png', img, img_size=imgsz, stride=32)
            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.float()
                # img = img.half() if half else img.float()  # uint8 to fp16/32
                # print("one", img.shape, type(img))
                # cv2.imwrite("tmp.png", np.array(img.cpu())) 
                # print("onepfive", img.shape, type(img))
                # img = img.reshape(())
                img /= 255.0  # 0 - 255 to 0.0 - 1.0

                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # img = img.permute(0, 3, 1, 2)
                # print("two", img.shape, type(img))
                # Inference
                t1 = time_synchronized()
                [pred,anchor_grid],seg,ll= model(img)
                t2 = time_synchronized()

                # print("img works")
                # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version 
                # but this problem will not appear in offical version 
                # TODO: maybe we can remove? Need to test with CUDA
                tw1 = time_synchronized()
                pred = split_for_trace_model(pred,anchor_grid)
                tw2 = time_synchronized()

                da_seg_mask = driving_area_mask(seg)
                ll_seg_mask = lane_line_mask(ll)
                # ll_seg_mask[ll_seg_mask == 1] = 255
                # cv2.imwrite("tmp1.png", ll_seg_mask) 
                s = '%gx%g ' % img.shape[2:]
                # Print time (inference)
                print(f'{s}Done. ({t2 - t1:.3f}s)')
                if self.save_imgs:
                    save_masks(da_seg_mask.copy(), ll_seg_mask.copy(), 'runs/detect')  
            
                # inf_time.update(t2-t1,img.size(0))
                # waste_time.update(tw2-tw1,img.size(0))
                # print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
                print(f'Done. ({time.time() - t0:.3f}s)')
            
            return da_seg_mask, ll_seg_mask
        except Exception as e:
            print("lane detection error:", e)
            return None, None


if __name__ == '__main__':
    # init args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = torch.jit.load('data/weights/yolopv2.pt', map_location="cpu")
    
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector(model, device, save_imgs=False)
    
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)
