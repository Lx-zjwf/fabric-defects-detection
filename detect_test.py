from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.apis import show_result
import os
from time import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# 模型配置文件
config_file = 'cascade_rcnn_r50_fpn_1x.py'

# 预训练模型文件
checkpoint_file = 'work_dirs/dcn/epoch_25.pth'

# 通过模型配置文件与预训练文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 测试单张图片
base_route = "/home/ubuntu/Project/fabric_defects_detection/fabric datast/"
img = base_route + 'coco/images/test/f7fd51f8598d61941044021532.jpg'
# cv2.inshow("fabric image", img)
# 统计检测时间
start = time()
result = inference_detector(model, img)
end = time()
print("run time: "+str(end-start))
score_thr = 0.10
show_result(img, result, model.CLASSES, score_thr)