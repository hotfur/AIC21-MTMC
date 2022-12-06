#!/bin/bash

seqs=(c041 c042 c043 c044 c045 c046)
# seqs=(c041)
for seq in ${seqs[@]}
do
    CUDA_VISIBLE_DEVICES="0,1" python detect.py --name ${seq} --weights yolov7-e6e.pt --conf 0.1 --agnostic --save-txt --save-conf --img-size 1280 --classes 2 5 7 --cfg_file $1&
    wait
done
wait
