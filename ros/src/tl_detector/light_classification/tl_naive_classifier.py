from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import rospy

class TLNaiveClassifier(object):
    def __init__(self):
        # your model should be able to handle all resolution
        # define your input resolution here
        # so that image can be resized accordingly
        self.width = 0
        self.height = 0
        self.channels = 3
        # create your model here, with tf, keras, scikitlearn or whatever you wish
        self.model = None
        self.red_cnt = 0

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # resized = cv2.resize(image, (self.width,self.height))
        # resized = resized / 255.; # Normalization

        # color = self.model.predict(resized.reshape((1, self.height, self.width, self.channels)))
        # tl = TrafficLight()
        # tl.state = color
        # return tl.state
        red_mask = (image[:,:,0]>210)&(image[:,:,1]<30)&(image[:,:,2]<30)
        self.red_cnt = np.sum(red_mask)
        if self.red_cnt>1:
            return TrafficLight.RED

        yellow_mask = (image[:,:,0]>210)&(image[:,:,1]>210)&(image[:,:,2]<30)
        self.yellow_cnt = np.sum(yellow_mask)
        if self.yellow_cnt>1:
            return TrafficLight.YELLOW

        return TrafficLight.GREEN
