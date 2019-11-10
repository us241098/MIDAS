#!/bin/bash

conda create -n emotion_detect python=3.7 -y
source activate emotion_detect
pip install tensorflow, keras, numpy, pandas,matplotlib, librosa

