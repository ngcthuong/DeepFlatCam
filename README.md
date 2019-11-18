# DeepFlatCam
Deep Compressive Sensing for Visual Privacy Protection in FlatCam Imaging (ICCV Workshop)

## Abstract 

Detection followed by projection in conventional privacy cameras is vulnerable to software attacks that threaten to expose image sensor data. By multiplexing the incoming light with a coded mask, a FlatCam camera removes the spatial correlation and captures visually protected images. However, FlatCam imaging suffers from poor reconstruction quality and pays no attention to the privacy of visual information. In this paper, we propose a deep learning-based compressive sensing approach to reconstruct and protect sensitive regions from secured FlatCam measurements. We predict sensitive regions via facial segmentation and separate them from the captured measurements. Our deep compressive sensing network was trained with simulated data, and was tested on both simulated and real FlatCam data.

## Source code
Training code in Pytorch 1.04 for Phase 1 and Phase 2 are in src1, and src2, respectively. 
Training data is used with DIK2000 dataset follow the instruction in ERSGAN 

## Document
Poster and paper are availble at "doc" folders. 

## For Citation 
If you find this source code, paper useful, please cite our work with

@InProceedings{Canh_2019_ICCV_Workshops,
author = {Nguyen Canh, Thuong and Nagahara, Hajime},
title = {Deep Compressive Sensing for Visual Privacy Protection in FlatCam Imaging},
booktitle = {The IEEE International Conference on Computer Vision (ICCV) Workshops},
month = {Oct},
year = {2019}
}
