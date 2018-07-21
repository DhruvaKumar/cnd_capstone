from styx_msgs.msg import TrafficLight
import rospy
import numpy as np
import os
import tensorflow as tf
from time import time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
# from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

NUM_CLASSES = 4
SCORE_THRESHOLD = 0.6

SAVE_OUTPUT_IMAGES = False

LABEL_MAP= {
    1: 'Red',
    2: 'Yellow',
    3: 'Green',
    4: 'Unknown'
}
COLOR_CODE_MAP = {
    1: (255,0,0),
    2: (255,255,0),
    3: (0,255,0),
    4: (255,255,255)
}


class TLClassifier(object):
    def __init__(self):
        
        # get model path
        self.model_path = rospy.get_param("/model_path")
        rospy.loginfo('tl_classifier: model_path:{}'.format(self.model_path))
        
        # load frozen graph
        graph = self.load_graph()

        # load tf session
        self.sess = tf.Session(graph=graph)

        # get tensors
        self.image_tensor = graph.get_tensor_by_name('image_tensor:0')
        self.num_detections_tnsr = graph.get_tensor_by_name('num_detections:0')
        self.classes_tnsr = graph.get_tensor_by_name('detection_classes:0')
        # self.boxes_tnsr = graph.get_tensor_by_name('detection_boxes:0')
        self.scores_tnsr = graph.get_tensor_by_name('detection_scores:0')

        rospy.loginfo('tl_classifier: loaded model and tf session')
        
        if SAVE_OUTPUT_IMAGES:
            self.bridge = CvBridge()
            self.img_op_pub = rospy.Publisher('/image_op', Image, queue_size=1)
        

    def load_graph(self):
        '''
        loads frozen graph from MODEL_PATH
        '''
        graph = tf.Graph()
        with graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(self.model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        return graph

    def get_classification(self, image, true_state):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        start = time()

        state = TrafficLight.UNKNOWN

        # run inference
        [num_detections, classes, scores] = self.sess.run([
            self.num_detections_tnsr, self.classes_tnsr, self.scores_tnsr],
            feed_dict={self.image_tensor: image[None,:]})
        
        num_detections =  int(num_detections)
        classes = classes[0].astype(np.uint8)
        # boxes = boxes[0]
        scores = scores[0]

        # max score
        max_score_id = np.argmax(scores)
        max_score = scores[max_score_id]
        class_max_score = classes[max_score_id]
        
        # return class of max score
        if (num_detections > 0 and max_score > SCORE_THRESHOLD):
            if (class_max_score == 1):
                state = TrafficLight.RED
            elif (class_max_score == 2):
                state = TrafficLight.YELLOW
            elif (class_max_score == 3):
                state = TrafficLight.GREEN

        rospy.logdebug('tl_classifier: detection took {:.2f}ms state:{} tstate:{}'.format((time() - start)*1000, state, true_state))
        
        
#         # debug misclassification
#         if state != true_state:
#             cv2.imwrite('misclassifications/{}_{}_{}_{:.2f}.png'.format(state, true_state, class_max_score, max_score), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#             rospy.logdebug('tl_classifier: saving misclassified image...')
            
        # debug op images
        if SAVE_OUTPUT_IMAGES:
            # overlay traffic state on image and publish to /image_op
            text = "None"
            font_color = (255,255,255)
            if state != TrafficLight.UNKNOWN:
                text = "{}: {:.2f}%".format(LABEL_MAP[class_max_score], 100*max_score)
                font_color = COLOR_CODE_MAP[class_max_score]
            font_scale = 2
            line_type = 2
            rows, cols, _ = image.shape
            cv2.putText(image,
                        text, 
                        (50,rows-100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale,
                        font_color,
                        line_type)
            # publish
            msg_image_op = self.bridge.cv2_to_imgmsg(image, "rgb8")
            self.img_op_pub.publish(msg_image_op)
        
        return state
