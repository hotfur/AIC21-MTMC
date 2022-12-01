#!/bin/bash

conda activate ML_Synh_prj
MCMT_CONFIG_FILE="aic_all.yml"
#### Run Detector.####
cd detector/
python gen_images_aic.py ${MCMT_CONFIG_FILE}
cd yolov5/
bash gen_det.sh ${MCMT_CONFIG_FILE}

