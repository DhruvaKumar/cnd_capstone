from styx_msgs.msg import TrafficLight
import numpy as np
import os.path
import sys
from glob import glob
import tensorflow as tf
from time import time

# from matplotlib import pyplot as plt

NUM_CLASSES = 4
MODEL_PATH = os.path.join('light_classification', 'models', 'frozen_inference_graph_ssdcoco.pb')


class TLClassifier(object):
    def __init__(self):
        
        # load frozen graph
        self.graph = self.load_graph()
        print('TLClassifier: loaded model')

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

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        start = time()

        state = TrafficLight.UNKNOWN

        with tf.Session(graph=self.graph) as sess:

            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes'
              ]:
              tensor_name = key + ':0'
              if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            
    #       print(tensor_dict)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: image[None,:]})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            
            if (output_dict['num_detections'] > 0):

                # return class of max score
                max_score_id = np.argmax(output_dict['detection_scores'])
                if (output_dict['detection_classes'][max_score_id] == 0):
                    state = TrafficLight.RED
                elif (output_dict['detection_classes'][max_score_id] == 1):
                    state = TrafficLight.YELLOW
                elif (output_dict['detection_classes'][max_score_id] == 2):
                    state = TrafficLight.GREEN

            print('tl_classifier: detection took {:.3f}s state:{}'.format(time() - start, state))

        return state
