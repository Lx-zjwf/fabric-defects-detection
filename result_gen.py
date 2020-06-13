import time, os
import json
from mmdet.apis import init_detector, inference_detector

# 使用指定的GPU运行程序
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    config_file = 'cascade_rcnn_r50_fpn_1x.py'
    checkpoint_file = 'work_dirs/benchmark/epoch_30.pth'

    test_path = 'coco/images/test'  # 官方测试集图片路径

    json_name = "result_" + "" + time.strftime("%Y%m%d%H%M%S", time.localtime()) + ".json"

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    img_list = []
    for img_name in os.listdir(test_path):
        if img_name.endswith('.jpg'):
            img_list.append(img_name)

    result = []
    for i, img_name in enumerate(img_list, 1):
        full_img = os.path.join(test_path, img_name)
        predict = inference_detector(model, full_img)

        # 比较读取到的配置信息与实际检测信息的差距
        for i, bboxes in enumerate(predict, 1):
            if len(bboxes) > 0:
                defect_label = i
                print(i)
                image_name = img_name
                for bbox in bboxes:
                    x1, y1, x2, y2, score = bbox.tolist()
                    x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)  # 保留两位有效数字

                    result.append(
                        {'name': image_name, 'category': defect_label, 'bbox': [x1, y1, x2, y2], 'score': score})

    with open(json_name, 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    main()