import os

# 构建瑕疵字典
defect_label2name = {
    1: '破洞', 2: '污渍', 3: '三丝', 4: '结头', 5: '花板跳', 6: '百脚', 7: '毛粒', 8: '粗经',
    9: '松经', 10: '断经', 11: '吊经', 12: '粗维', 13: '纬缩', 14: '浆斑', 15: '整经结',
    16: '跳花', 17: '断氨纶', 18: '色差档', 19: '磨痕', 20: '死皱'
}
defect_name2label = {
    '破洞': 1, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7, '粗经': 8,
    '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15,
    '跳花': 16, '断氨纶': 17, '色差档': 18, '磨痕': 19, '死皱': 20
}

gt_root = '/home/ubuntu/Project/fabric_defects_detection/detect demo/mAP/input/ground-truth'
gt_file_list = os.listdir(gt_root)
dr_root = '/home/ubuntu/Project/fabric_defects_detection/detect demo/mAP/input/detection-results'
dr_file_list = os.listdir(dr_root)

iou_sum = [0.0] * 20
iou_num = [0.0] * 20
defect_sum = 0.0  # 瑕疵总数
detect_sum = 0.0  # 检测框总数
true_detect_sum = 0.0  # 正确的检测框数量
true_defect_sum = 0.0
con_thre = 0.0
iou_thre = 0.45

# 依次比较相同文件的内容
for gt_path in gt_file_list:
    gt_file = os.path.join(gt_root, gt_path)
    dr_file = os.path.join(dr_root, gt_path)
    # 按行读取文件内容
    with open(gt_file, 'r') as f:
        gt_info = f.readlines()
    with open(dr_file, 'r') as f:
        dr_info = f.readlines()

    for gt in gt_info:
        defect_sum += 1
        gt_str = gt.split(" ")
        gt_index = defect_name2label[gt_str[0]]  # 瑕疵的索引
        max_iou = 0.0  # 设置iou的最大重叠值
        gt_coord = [float(gt_str[1]), float(gt_str[2]),
                    float(gt_str[3]), float(gt_str[4])]
        # if(gt_str[0] == "三丝"):
        #     print(gt_path)
        # 未检测到瑕疵
        if dr_info == []:
            iou_num[gt_index-1] += 1
            continue

        for dr in dr_info:
            dr_str = dr.split(" ")
            if(float(dr_str[1]) < con_thre):
                continue
            if dr_str[5][-1:] == '\n':
                dr_str[5] = dr_str[5][:-1]

            dr_coord = [float(dr_str[2]), float(dr_str[3]),
                        float(dr_str[4]), float(dr_str[5])]
            # 定义交集和并集的坐标
            inter_coord = []
            inter_coord.append(max(gt_coord[0], dr_coord[0]))
            inter_coord.append(max(gt_coord[1], dr_coord[1]))
            inter_coord.append(min(gt_coord[2], dr_coord[2]))
            inter_coord.append(min(gt_coord[3], dr_coord[3]))
            # 计算iou
            inter_area = (inter_coord[2] - inter_coord[0]) * (inter_coord[3] - inter_coord[1])
            union_area = (dr_coord[2] - dr_coord[0]) * (dr_coord[3] - dr_coord[1]) + \
                         (gt_coord[2] - gt_coord[0]) * (gt_coord[3] - gt_coord[1]) - inter_area

            if ((inter_coord[2] - inter_coord[0]) < 0) | ((inter_coord[3] - inter_coord[1]) < 0):
                iou = 0.0
            else:
                iou = inter_area / union_area

            if iou > max_iou:
                max_iou = iou

        if max_iou > iou_thre:
            true_defect_sum += 1

        iou_sum[gt_index - 1] += max_iou
        iou_num[gt_index - 1] += 1

    gt_num = len(gt_info)  # 目标的数量
    gt_detect = [-1] * gt_num  # 设置所有目标都未被检测情况
    for dr in dr_info:
        # 未检测到瑕疵
        if dr_info == []:
            continue
        dr_str = dr.split(" ")
        if (float(dr_str[1]) < con_thre):
            continue
        if dr_str[5][-1:] == '\n':
            dr_str[5] = dr_str[5][:-1]

        dr_coord = [float(dr_str[2]), float(dr_str[3]),
                    float(dr_str[4]), float(dr_str[5])]

        max_iou = 0.0
        gt_label = -1  # 与检测框交并比最大的真值框编号
        for i, gt in enumerate(gt_info):
            gt_str = gt.split(" ")
            gt_coord = [float(gt_str[1]), float(gt_str[2]),
                        float(gt_str[3]), float(gt_str[4])]
            # 定义交集和并集的坐标
            inter_coord = []
            inter_coord.append(max(gt_coord[0], dr_coord[0]))
            inter_coord.append(max(gt_coord[1], dr_coord[1]))
            inter_coord.append(min(gt_coord[2], dr_coord[2]))
            inter_coord.append(min(gt_coord[3], dr_coord[3]))
            # 计算iou
            inter_area = (inter_coord[2] - inter_coord[0]) * (inter_coord[3] - inter_coord[1])
            union_area = (dr_coord[2] - dr_coord[0]) * (dr_coord[3] - dr_coord[1]) + \
                         (gt_coord[2] - gt_coord[0]) * (gt_coord[3] - gt_coord[1]) - inter_area

            if ((inter_coord[2] - inter_coord[0]) < 0) | ((inter_coord[3] - inter_coord[1]) < 0):
                iou = 0.0
            else:
                iou = inter_area / union_area

            if iou > max_iou:
                max_iou = iou
                gt_label = i

        if gt_detect[gt_label] == -1:  # 该瑕疵未被检测过
            detect_sum += 1
            if max_iou > iou_thre:
                true_detect_sum += 1
                gt_detect[gt_label] = 1

# 显示最终的检测结果
sum_iou = 0.0
for i in range(20):
    print(defect_label2name[i+1]+":")
    print(round(iou_sum[i]/iou_num[i]*100.0, 2))
    sum_iou += iou_sum[i] / iou_num[i]
average_iou = sum_iou / 20.0 * 100.0
precision = true_detect_sum / detect_sum * 100.0
recall = true_defect_sum / defect_sum * 100.0
print("average iou: ", round(average_iou, 2))
print("precision: ", round(precision, 2))
print("recall: ", round(recall, 2))
print("F1 score:", round((2*precision*recall)/(precision+recall), 2))