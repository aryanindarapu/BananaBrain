import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String, Float64
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PILImage
import torch
from matplotlib import pyplot as plt
from matplotlib import patches

class stopsign_detector:
    def __init__(self, image_processor, model, device):
        self.device = device
        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        # self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        # self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)
        # TODO: change the topic names for left and right cameras
        self.zed_sub_img = rospy.Subscriber('zed2/zed_node/rgb/image_rect_color', Image, self.sign_detection, queue_size=1)
        
        # NOTE: this currently runs every second, want to make duration smaller
        # self.timer = rospy.Timer(rospy.Duration(1), self.process_cameras)
        self.obj_detection_pub = rospy.Publisher('stop_sign/object_detection', Image, queue_size=1)
        self.stop_sign_pub = rospy.Publisher('stop_sign/score', Float64, queue_size=1)

        self.image_processor = image_processor
        self.model = model
        self.model.to(device)

    def sign_detection(self, img):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
            raw_img = cv_image.copy()
            tmp_img = raw_img.copy()
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            image = PILImage.fromarray(raw_img)
            
            inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            
            target_sizes = torch.tensor([image.size[::-1]])
            results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
            
            stop_sign = None
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if label.item() == 13 and score.item() > 0.5:
                    stop_sign = box
                    box = [round(i, 2) for i in box.tolist()]
                    
                    cv2.rectangle(tmp_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

                    print(
                        f"Detected {model.config.id2label[label.item()]} with confidence "
                        f"{round(score.item(), 3)} at location {box}"
                    )
            
            self.obj_detection_pub.publish(self.bridge.cv2_to_imgmsg(tmp_img, "bgr8"))

            # determine how much of image stop sign is taking up
            if stop_sign is not None:
                box_area = (stop_sign[2] - stop_sign[0]) * (stop_sign[3] - stop_sign[1])
                image_area = raw_img.shape[0] * raw_img.shape[1]
                print("Score", float(box_area / image_area) * 10)
                self.stop_sign_pub.publish(Float64(float(box_area / image_area) * 10))
            else:
                print("Score", 0.0)
                self.stop_sign_pub.publish(Float64(0.0))
            
               
        except Exception as e:
            print("stop sign detection error:", e)
            

if __name__ == '__main__':
    # init args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")
    model = AutoModelForObjectDetection.from_pretrained("microsoft/conditional-detr-resnet-50")

    rospy.init_node('stopsign_node', anonymous=True)
    stopsign_detector(image_processor, model, device)
    
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)
