#!/bin/sh

# create models dir
mkdir -p src/tl_detector/light_classification/models
cd src/tl_detector/light_classification/models 

# if using new models, make sure to update the launch files in tl_detector

# download sim model
wget -N https://s3.amazonaws.com/dhruvak/frozen_inference_graph_ssdcoco.pb

# download site model
wget -N https://s3.amazonaws.com/dhruvak/frozen_inference_graph_fasterrcnn_real.pb