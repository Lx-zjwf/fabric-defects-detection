# fabric-defects-detection
A detection model for fabric defects based on deep Cascade R-CNN is proposed, focusing on the substantive difficulties such as the increase of hard examples caused by various patterns and varieties of defects.<br>

# Requirements
Python3.7, Pytorch 1.1.0 and other comman packages listed in requirements.txt.<br>
Install and compile mmdetection from <https://github.com/open-mmlab/mmdetection>.<br>
Download project from <https://github.com/Cartucho/mAP> to caluculate mAP.<br>
The fabric datasets and pretrained weight are contained in <https://pan.baidu.com/s/1TDBU5a5WPtr9kAdIIkvM-Q>, and the password is tlfz. You should unzip the packages to corrsponding folder, and merge two train datasets into one folder.

# Instruction
Before training, you should modify parameters in config file cascade_rcnn_r50_fpn_1x.py, the paths data_root, ann_file, img_prefix, work_dir and load_from should be set by yourself. Run mmdetection_train.py to train network and mmdetection_test or detect_test to test. To calculate mAP, you should first run detect_data.py to generate ground truth file and detection result file. The you can replace folder "ground-truth" and "detection-results" in project "mAP", run index_calc.py and mAP_calc.py to calculate the evaluation index.
