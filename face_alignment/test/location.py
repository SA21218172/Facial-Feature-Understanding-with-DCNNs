import sys 
sys.path.append(".") 
import face_alignment
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import json

os.environ ["CUDA_VISIBLE_DEVICES"] = '7'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

def get_bounding_box(part_array):
    x, y = part_array[0][0],part_array[0][1]
    left = right = x
    up = down = y
    for i in range(len(part_array)):
        temp_x, temp_y = part_array[i][0],part_array[i][1]
        if(temp_x<left):
            left = temp_x
        if(temp_x>right):
            right = temp_x
        if(temp_y<up):
            up = temp_y
        if(temp_y>down):
            down = temp_y

    ans = [int(left), int(right), int(up), int(down)]
    return ans
def get_same_area_box(origin_box, area=400):
    #返回相同origin_box比例的指定面积的box
    left, right, up, down = origin_box
    width = right - left
    height = down - up
    new_width = int(np.sqrt(width*area/height))
    # new_height = int(area/new_width)
    new_height = int(np.sqrt(height*area/width))
    new_left = int(left + (width-new_width)/2)
    new_right = int(right + (new_width-width)/2)
    new_up = int(up + (height-new_height)/2)
    new_down = int(down + (new_height - height)/2)
    box = [new_left, new_right, new_up, new_down]
    # print(origin_box, box)
    return box


def loc(preds, image_name):
    result = {}
    local = {}
    result[image_name] = local
    # 获得眼睛的loc
    # get left eyes_box
    left_eye = preds[0][36:42]
    eye_box_1 = get_bounding_box(part_array=left_eye)
    # get right eyes_box
    right_eye = preds[0][42:48]
    eye_box_2 = get_bounding_box(right_eye)
    # eyes_box_1 = get_same_area_box(eye_box_1)
    # eyes_box_2 = get_same_area_box(eye_box_2)
    eyes = eye_box_1 + eye_box_2
    local["eyes"] = eyes

    # 获得eyebrow的loc
    left_eyebrow = preds[0][17:22]
    eyebrow_box_1 = get_bounding_box(part_array=left_eyebrow)
    right_eyebrow = preds[0][22:27]
    eyebrow_box_2 = get_bounding_box(part_array=right_eyebrow)
    # eyebrow_box_1 = get_same_area_box(eyebrow_box_1, area=200)
    # eyebrow_box_2 = get_same_area_box(eyebrow_box_2, area=200)
    local["eyebrow"] = eyebrow_box_1 + eyebrow_box_2

    # 获得mouth的loc
    mouth = preds[0][49:68]
    mouth_box= get_bounding_box(mouth)
    # mouth_box = get_same_area_box(mouth_box)
    local["mouth"] = mouth_box

    # 获得nose的loc
    nose = preds[0][27:36]
    nose_box = get_bounding_box(nose)
    # nose_box = get_same_area_box(nose_box)
    local["nose"] = nose_box
    # print(result)
    return result

if __name__ =="__main__":
    # image_path = "/home/qianqian/dataset/test_pic/40.jpg"
    # image_name = image_path[image_path.rfind('/')+1:]
    # input = cv2.imread(image_path)
    # preds = fa.get_landmarks(input)
    # result = loc(preds=preds, image_name=image_name)

    ori_path = "/home/qianqian/dataset/face_data_3_val/"
    folder_list = os.listdir(ori_path)
    result = {}
    i = 0
    for folder in folder_list:
        folder_path = os.path.join(ori_path, folder)
        image_list = os.listdir(folder_path)
        for image in image_list:
            image_path = os.path.join(folder_path, image)
            image_name = image_path[image_path.rfind('/')-5:]
            input = cv2.imread(image_path)
            preds = fa.get_landmarks(input)
            try:
                temp_dict = loc(preds, image_name)
            except ZeroDivisionError:
                print(image_path, "zero error")
            except TypeError:
                print(image_path, "Type error")
            result = dict(result, **temp_dict)
            i += 1
            if i % 100 == 0:
                print(i)


    # 写入json文件
    file_name = "./face_3_loc_indices_not_same_area.json"
    json_str = json.dumps(result,indent=4)
    with open(file_name, 'w') as json_file:
        json_file.write(json_str)




