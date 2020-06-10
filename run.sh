#!/bin/bash
echo "waiting for 1 hour..."
sleep 1h


python train.py --root /Volumes/IPEVO_X0244/coco_dataset/ --pretrain-model /Users/kehwaweng/Documents/ObjectDetection/torch_ssd_mobilenet/experience/20200505_0947/model_weights_checkpoint_epoch_99.pt --image-size 300 -lr 4e-6

