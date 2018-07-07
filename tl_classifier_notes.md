# Traffic light classifier notes

### goals

- get tl classification working in simulator
- get tl classification working with real world data (udacity rosbag)
- get tl classification working at test site


### training data

- sim data https://drive.google.com/drive/u/0/folders/0BzcEGp8MN5DidEZFbG8wbnQzNEU
	- 280 annotated images 
	- contains images + yml file detailing bounding box positions and states
	- posted on slack by anthony sarkis 

- sim data https://github.com/level5-engineers/system-integration/wiki/Traffic-Lights-Detection-and-Classification
	- 5k+ images
	- does not contain bounding boxes. just contains images and corresponding class (red,orage,green,unknown)

- rosbag udacity https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip
	- contains images taken from carla at test site

- bosch 
https://hci.iwr.uni-heidelberg.de/node/6132
https://github.com/bosch-ros-pkg/bstld

- lisa traffic light dataset
http://cvrr.ucsd.edu/vivachallenge/index.php/traffic-light/
http://cvrr.ucsd.edu/vivachallenge/index.php/traffic-light/traffic-light-detection/

- LaRa traffc light recognition
http://www.lara.prd.fr/benchmarks/trafficlightsrecognition

### test data

https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing


## references

- anthony sarkis
https://codeburst.io/self-driving-cars-implementing-real-time-traffic-light-detection-and-classification-in-2017-7d9ae8df1c58

- vatsal srivastava
https://becominghuman.ai/traffic-light-detection-tensorflow-api-c75fdbadac62

- interesting analysis by diyjac
https://github.com/diyjac/SDC-System-Integration/tree/master/classifier

- David Brailovsky's post. Hhe won the Nexar Traffic light challenge
https://medium.freecodecamp.org/recognizing-traffic-lights-with-deep-learning-23dae23287cc

- https://github.com/level5-engineers/system-integration/wiki/Traffic-Lights-Detection-and-Classification


- traffic light mapping and detection by nathaniel fairfield and chris urmson, google https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37259.pdf

- Speed/accuracy trade-offs for modern convolutional object detectors https://arxiv.org/abs/1611.10012