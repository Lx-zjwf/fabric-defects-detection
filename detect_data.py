# 读取测试集的标注信息并生成用于mAP计算的文件
import json
import time, os
import json
from mmdet.apis import init_detector, inference_detector

defect_label2name = {
    1: '破洞', 2: '污渍', 3: '三丝', 4: '结头', 5: '花板跳', 6: '百脚', 7: '毛粒', 8: '粗经',
    9: '松经', 10: '断经', 11: '吊经', 12: '粗维', 13: '纬缩', 14: '浆斑', 15: '整经结',
    16: '跳花', 17: '断氨纶', 18: '色差档', 19: '磨痕', 20: '死皱'
}

def main():

    # # 存储标注信息
    # json_file = '/home/ubuntu/Project/fabric_defects_detection/fabric datast/' \
    #             'coco/annotations/instances_train.json'
    # cfg_info = json.load(open(json_file, 'r'))
    #
    # anno_info = cfg_info["annotations"]  # 读取标注信息
    # image_info = cfg_info["images"]  # 读取图像信息
    #
    # # 设置初始化信息
    # image_id = -1
    #
    # for anno_dict in anno_info:
    #     cur_id = anno_dict["image_id"]  # 读取当前图像的id
    #     output = ""  # 写入文本中的内容
    #     # 判断是否还是上一组信息对应的图像
    #     if cur_id != image_id:
    #         image_id = cur_id
    #         image_name = image_info[image_id]["file_name"]
    #         file_name = image_name.split(".", 1)[0]  # 提取图像名称
    #         text_name = file_name + ".txt"  # 对应的文本文件信息
    #     else:
    #         output = "\n"
    #
    #     output += defect_label2name[anno_dict["category_id"]] + " "
    #     output += str(anno_dict["bbox"][0]) + " " + str(anno_dict["bbox"][1]) + " " + \
    #               str(anno_dict["bbox"][0] + anno_dict["bbox"][2]) + " " + str(
    #         anno_dict["bbox"][1] + anno_dict["bbox"][3])
    #
    #     # 将信息写入文本中
    #     text_path = "ground-truth/" + text_name
    #     f = open(text_path, 'a')
    #     f.write(output)
    #     f.close()

    # 存储测试结果
    folder_name = 'work_dirs/ohem+focal loss0.5/'
    config_file = 'cascade_rcnn_r50_fpn_1x.py'
    checkpoint_file = folder_name + 'epoch_50.pth'

    test_path = '/home/ubuntu/Project/fabric_defects_detection/' \
                'fabric datast/coco/images/test'

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    image_list = []
    for image_name in os.listdir(test_path):
        if image_name.endswith('.jpg'):
            image_list.append(image_name)

    result = []
    for i, image_name in enumerate(image_list, 1):
        full_img = os.path.join(test_path, image_name)
        predict = inference_detector(model, full_img)

        file_name = image_name.split(".", 1)[0]  # 提取图像名称
        text_name = file_name + ".txt"  # 对应的文本文件信息
        output = ""

        # 比较读取到的配置信息与实际检测信息的差距
        for i, bboxes in enumerate(predict, 1):
            if len(bboxes) > 0:
                defect_label = i
                for bbox in bboxes:
                    x1, y1, x2, y2, score = bbox.tolist()
                    x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)  # 保留两位有效数字
                    output += defect_label2name[defect_label] + " " + str(score) + " " + str(x1) + " "\
                          + str(y1) + " " + str(x2) + " " + str(y2)
                    output += "\n"

        # 将信息写入文本中
        text_path = folder_name + "detection-results/" + text_name
        f = open(text_path, 'a')
        f.write(output)
        f.close()

if __name__ == "__main__":
    main()