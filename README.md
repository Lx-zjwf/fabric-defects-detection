# fabric-defects-detection
A detection model for fabric defects based on deep Cascade R-CNN is proposed, focusing on the substantive difficulties such as the increase of hard examples caused by various patterns and varieties of defects.<br>

# Requirements
Python3.7, Pytorch 1.1.0 and other comman packages listed in requirements.txt.<br>
Install and compile mmdetection from <https://github.com/open-mmlab/mmdetection>.<br>
Download project from <https://github.com/Cartucho/mAP> to caluculate mAP.<br>
The fabric datasets and pretrained weight are contained in <https://pan.baidu.com/s/1TDBU5a5WPtr9kAdIIkvM-Q>, and the password is tlfz. You should unzip the packages to corrsponding folder, and merge two train datasets into one folder.

# Instruction
Before training, you should modify parameters in config file cascade_rcnn_r50_fpn_1x.py according your needs. The paths such as data_root, ann_file, img_prefix, work_dir and load_from should be set by yourself. Run mmdetection_train.py to train network and mmdetection_test or detect_test to test. To calculate mAP, you should first run detect_data.py to generate ground truth file and detection result file. Then you can replace the folders "ground-truth" and "detection-results" in project "mAP", run index_calc.py and mAP_calc.py to calculate evaluation indexes.

# Introduction
This item provide a Cascade R-CNN for fabric defects detection, based on the baseline provided by sloan and jianh <https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.9.43b46448UDpstK&postId=74264> and datasets provided by Tianchi algorithm Contest <https://tianchi.aliyun.com/competition/entrance/231748/introduction?spm=5176.12281957.1004.5.38b02448NiKTgT>. The Resnet50 is set as backbone network to extract feature, combined with feature pyramid network and deformable convolution. We use three 
R-CNN branches to locate and classify defects with gradual increasing iou threshold, and each branch samples by OHEM algorithm. The model can obtain 61.67% average iou, 78.67% precision, 79.79% recall and 62% mAP. The detection results can be shown as follows:

APs of detection results in each defect class are illustrated in the following picture: 
