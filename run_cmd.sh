#!/bin/bash

bash /nfs/volume-902-16/tangwenbo/ofs-1.sh

cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/MiniGPT-4 && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
  command.py
