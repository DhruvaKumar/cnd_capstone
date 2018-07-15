from styx_msgs.msg import TrafficLight
import rospy
import numpy as np
import os
import sys
from glob import glob
import tensorflow as tf
from time import time

import cv2
# from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

NUM_CLASSES = 4
MODEL_PATH = os.path.join('light_classification', 'models', 'frozen_inference_graph_ssdcoco.pb')
SCORE_THRESHOLD = 0.6


class TLClassifier(object):
    def __init__(self):
        
        # load frozen graph
        self.graph = self.load_graph()
        rospy.logdebug('TLClassifier: loaded model')
        
        self.count_save = 0

    def load_graph(self):
        '''
        loads frozen graph from MODEL_PATH
        '''
        graph = tf.Graph()
        with graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
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

        with tf.Session(graph=self.graph) as sess:
            
            # get tensors
            num_detections_tnsr = tf.get_default_graph().get_tensor_by_name('num_detections:0')
            classes_tnsr = tf.get_default_graph().get_tensor_by_name('detection_classes:0')
#             boxes_tnsr = tf.get_default_graph().get_tensor_by_name('detection_boxes:0')
            scores_tnsr = tf.get_default_graph().get_tensor_by_name('detection_scores:0')
        
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        
            # run inference
            [num_detections, classes, scores] = sess.run([
                num_detections_tnsr, classes_tnsr, scores_tnsr],
                feed_dict={image_tensor: image[None,:]})
            
            num_detections =  int(num_detections)
            classes = classes[0].astype(np.uint8)
#             boxes = boxes[0]
            scores = scores[0]

            # max score
            max_score_id = np.argmax(scores)
            max_score = scores[max_score_id]
            class_max_score = classes[max_score_id]
            
            if (num_detections > 0 and max_score > SCORE_THRESHOLD):

                # return class of max score
                if (class_max_score == 1):
                    state = TrafficLight.RED
                elif (class_max_score == 2):
                    state = TrafficLight.YELLOW
                elif (class_max_score == 3):
                    state = TrafficLight.GREEN

            rospy.logdebug('tl_classifier: detection took {:.3f}s state:{} tstate:{}'.format(time() - start, state, true_state))
            
            
            # debug misclassification
            if state != true_state:
                cv2.imwrite('misclassifications/{}_{}_{}_{:.2f}.png'.format(state, true_state, class_max_score, max_score), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                print('tl_classifier: saving misclassified image...')
#             if (self.count_save < 2):
#                 cv2.imwrite('{}_{}.png'.format(self.count_save, state), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#                 print(output_dict)
#                 self.count_save += 1

        return state
