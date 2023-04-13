import os
import json
import numpy as np
from iresnet_single_pic import extract_grayscale_cam
os.environ ["CUDA_VISIBLE_DEVICES"] = '3'

# 读取json文件内容,返回字典格式
with open('/home/qianqian/old_folder/data_from_1080ti/qianqian/face-alignment-master/face_2_loc_indices_not_same_area.json', 'r', encoding='utf8')as fp:
    json_data = json.load(fp)
    result = []
    sum_eye = sum_eyebrow = sum_mouth = sum_nose = 0.0
    i = 0
    nan = float('nan')
    for key, value in json_data.items():
        avg_temp = extract_grayscale_cam(key, value)
        result.append(avg_temp)
        if np.isnan(result[i][0]) or np.isnan(result[i][1]) or np.isnan(result[i][2]) or np.isnan(result[i][3]):
            print("err is {}".format(key))
            i += 1
            continue
        sum_eye += result[i][0]
        sum_eyebrow += result[i][1]
        sum_mouth += result[i][2]
        sum_nose += result[i][3]
        if i%100 == 0:
            print([i, key, sum_eye, sum_eyebrow, sum_mouth, sum_nose])
        i += 1
    avg_eye = sum_eye/i
    avg_eyebrow = sum_eyebrow/i
    avg_mouth = sum_mouth/i
    avg_nose = sum_nose/i



    print([sum_eye, sum_eyebrow, sum_mouth, sum_nose], [avg_eye, avg_eyebrow, avg_mouth, avg_nose])
    print("face_3")
    r = {}
    r['record'] = [[sum_eye, sum_eyebrow, sum_mouth, sum_nose], [avg_eye, avg_eyebrow, avg_mouth, avg_nose]]
    r['result'] = result


    file_name = "./record_face_data_2_val_eyebrow.json"
    json_str = json.dumps(r,indent=4)
    with open(file_name, 'w') as json_file:
        json_file.write(json_str)



    # print(result)



