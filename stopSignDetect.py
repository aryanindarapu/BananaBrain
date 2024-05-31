import torch
import rospy

class stop_sign_detector:
    def __init__(self):
        self.model = torch.load('stopSignModel.pth')
        self.image = torch.load('image.pth')
        self.output = None

    def run(self):
        self.output = self.model(self.image)
        return self.output

    def publish_output(self):
        rospy.loginfo(self.output)

    def stop_sign_detect(self):
        self.run()
        self.publish_output()

if __name__ == '__main__':
    rospy.init_node('stopSignDetect')
    # Load the model
    model = torch.load('stopSignModel.pth')
    # Load the image
    image = torch.load('image.pth')
    # Run the model
    output = model(image)
    # Publish the output
    rospy.loginfo(output)
    rospy.spin()