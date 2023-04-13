# from turtle import right
from pickletools import uint8
from re import S
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import pdb
import sys
import copy
import random
from scipy.ndimage import filters
sys.path.append(".")
import face_alignment
FACIAL_LANDMARKS_68_IDXS = dict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
])

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

#input = io.imread('../test/assets/aflw-test.jpg')
# input = io.imread('../test/assets/1.jpg')
# preds = fa.get_landmarks(input)
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

def get_added_bounding_box(part_array):
    x, y = part_array[0][0], part_array[0][1]
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
    margin_1 = (right - left) * 0.1 # 两边同时扩大5%
    margin_2 = (down - up) * 0.3


    # 计算增加margin之后的坐标
    left = left - margin_1
    right = right + margin_1
    # print("up and margin_2 and down is {}, {}, {}".format(up, margin_2, down))
    up = up - margin_2
    down = down + margin_2
    ans = [int(left), int(right), int(up), int(down)]
    area = (right - left) * (down - up)
    # print("area is {}".format(area))
    return ans, area

def filp_part(input, box):
    #将前两个维度的box翻转,现在的问题是对于input来说，对应维度是高度或者宽度
    ans = input
    temp = input[box[2]:box[3], box[0]:box[1]]
    # plt.imshow(temp)
    # plt.show()
    temp_2 = cv2.flip(temp, 0)  #垂直翻转
    # plt.imshow(temp_2)
    # plt.show()
    #temp = temp.transpose(Image.FLIP_TOP_BOTTOM)  # 垂直翻转
    ans[box[2]:box[3], box[0]:box[1]] = temp_2
    return ans

def process_image(preds, input_image):

    # 处理左眼
    array_left_eye = preds[0][36:42]
    left_eye_box = get_bounding_box(array_left_eye)

    # 处理右眼
    array_right_eye = preds[0][42:48]
    right_eye_box = get_bounding_box(array_right_eye)

    # 处理嘴巴
    array_mouth = preds[0][49:68]
    mouth_box = get_bounding_box(array_mouth)

    #图片位置翻转
    filp_part(input_image, left_eye_box)
    filp_part(input_image,right_eye_box)
    filp_part(input_image,mouth_box)
    return  input_image


def get_value(input, box):
    mid = int((box[1]-box[0])/2 + box[0])
    if(box[3]>=112):
        print("box[3] is {}".format(box[3]))
        box[3] = 111
        pdb.set_trace()
    value = input[box[3], mid,:]
    return value
    

def mask_part(input, box):
    temp = copy.deepcopy(input)
    mask_value = get_value(input, box)
    for i in range(3):
        temp[box[2]:box[3], box[0]:box[1], i] = mask_value[i]
    
    im_blur = np.zeros(temp.shape)
    for i in range(3):
        im_blur[:,:,i] = filters.gaussian_filter(temp[:,:,i], sigma=10)
        #模糊完成后赋值到原矩阵
        input[box[2]:box[3], box[0]:box[1], i] = im_blur[box[2]:box[3], box[0]:box[1], i]

    return input

def get_rand_box(ori_pos):
    #传入的ori_pos是对应的四个点
    left, right, up, down = ori_pos
    width = right - left
    height = down - up
    x = random.randint(0, 112 - width)
    # print("h and down and up is {},{},{}".format(height, down, up))
    y = random.randint(0, 112 - height)
    return [x, x+width, y, y+height]


def get_same_area_box(origin_box, area:int):
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

def mask_eyes(preds, input_image,is_rand=False, is_the_same=False):
    # get left eyes_box
    #is_rand表示是否随机选择位置，在同样大小的面积的情况下,目前实现mask两眼之和的大小
    array_left_eye = preds[0][36:42]
    # left_eye_box, area = get_added_bounding_box(array_left_eye)
    left_eye_box = get_bounding_box(part_array=array_left_eye)
    # get right eyes_box
    array_right_eye = preds[0][42:48]
    # right_eye_box, area = get_added_bounding_box(array_right_eye)
    right_eye_box= get_bounding_box(array_right_eye)
    if not is_rand :
        if not is_the_same:
            mask_part(input_image, left_eye_box)
            mask_part(input_image, right_eye_box)
        else:
            #不随机mask但是mask掉相同面积的位置
            left_box = get_same_area_box(left_eye_box,area=400)
            right_box = get_same_area_box(right_eye_box, area=400)
            mask_part(input_image, left_box)
            mask_part(input_image, right_box)
    else:
        width_left = left_eye_box[1]-left_eye_box[0]
        h_left = left_eye_box[3]-left_eye_box[2]

        width_right = right_eye_box[1] - right_eye_box[0]
        h_right = right_eye_box[3] - right_eye_box[2]

        square = (width_left * h_left) + (width_right * h_right)
        # left, right, up, down
        box = get_rand_box([0,square//max(h_left, h_right), 0, max(h_left, h_right)])
        mask_part(input_image, box)

    return input_image

def mask_eyebrow(preds, input_image,is_rand=False, is_the_same=False):
    # get left eyes_box
    #is_rand表示是否随机选择位置，在同样大小的面积的情况下,目前实现mask两眼之和的大小
    array_left_eye = preds[0][17:22]
    # left_eye_box, area = get_added_bounding_box(array_left_eye)
    left_eye_box = get_bounding_box(part_array=array_left_eye)
    # get right eyes_box
    array_right_eye = preds[0][22:27]
    # right_eye_box, area = get_added_bounding_box(array_right_eye)
    right_eye_box= get_bounding_box(array_right_eye)
    if not is_rand :
        if not is_the_same:
            mask_part(input_image, left_eye_box)
            mask_part(input_image, right_eye_box)
        else:
            #不随机mask但是mask掉相同面积的位置
            left_box = get_same_area_box(left_eye_box,area=200)
            right_box = get_same_area_box(right_eye_box, area=200)
            mask_part(input_image, left_box)
            mask_part(input_image, right_box)
    else:
        width_left = left_eye_box[1]-left_eye_box[0]
        h_left = left_eye_box[3]-left_eye_box[2]

        width_right = right_eye_box[1] - right_eye_box[0]
        h_right = right_eye_box[3] - right_eye_box[2]

        square = (width_left * h_left) + (width_right * h_right)
        # left, right, up, down
        box = get_rand_box([0,square//max(h_left, h_right), 0, max(h_left, h_right)])
        mask_part(input_image, box)

    return input_image

def mask_mouth(preds, input_image, is_rand=False, is_the_same=False):
    array_mouth = preds[0][49:68]
    # mouth_box, area = get_added_bounding_box(array_mouth)
    mouth_box= get_bounding_box(array_mouth)
    if not is_rand:
        if not is_the_same:
            return mask_part(input_image, mouth_box)
        else:
            mouth_box = get_same_area_box(mouth_box, area=800)
            return mask_part(input_image, mouth_box)
    else:
        box = get_rand_box(mouth_box)
        return mask_part(input_image, box)

def mask_nose(preds, input_image, is_rand=False, is_the_same=False):
    array_nose = preds[0][27:36]
    # nose_box, area = get_added_bounding_box(array_nose)
    nose_box = get_bounding_box(array_nose)
    if not is_rand:
        if not is_the_same:
            return mask_part(input_image, nose_box)
        else:
            nose_box = get_same_area_box(nose_box, area=800)
            return mask_part(input_image, nose_box)
    else:
        box = get_rand_box(nose_box)
        return mask_part(input_image, box)
        
def visualize_facial_landmarks(image, shape, object_name, colors=None, alpha=0.75):
    # 创建两个copy
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()
    # 设置一些颜色区域
    if colors is None:
        colors = (0,0,0)
    # 遍历每一个区域
    for (i, name) in enumerate(FACIAL_LANDMARKS_68_IDXS.keys()):
        # 得到每一个点的坐标
        if name in object_name:
            (j, k) = FACIAL_LANDMARKS_68_IDXS[name]
            pts = shape[j:k]
            # 计算凸包
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull.astype(int)], -1, colors, -1)
    # 叠加在原图上，可以指定比例
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return overlay

def mask(image_path, mask_parts:list, save_path):
    #mask对应的轮廓
    input = cv2.imread(image_path)
    preds = fa.get_landmarks(input)[0]
    ans = visualize_facial_landmarks(input, preds, mask_parts)
    cv2.imwrite(save_path, ans)

def mask_box_add10(image_path, mask_parts:list, save_path, is_rand=False, is_the_same=False):
    # 返回的图像是扩大10%的box
    input = cv2.imread(image_path)
    preds = fa.get_landmarks(input)
    if "eyes" in mask_parts:
        ans = mask_eyes(preds, input, is_rand, is_the_same)
    if "eye_brow" in mask_parts:
        ans = mask_eyebrow(preds, input, is_rand, is_the_same)
    if "mouth" in mask_parts:
        ans = mask_mouth(preds, input, is_rand,is_the_same)
    if "nose" in mask_parts:
        ans = mask_nose(preds, input, is_rand, is_the_same)
    cv2.imwrite(save_path, ans)

# def mask_box_same(image_path, mask_parts:list, save_path, is_rand = Fasle):
#     #返回相同面积的mask的结果
#     input = cv2.imread(image_path)
#     preds = fa.get_landmarks(input)

if __name__ =="__main__":
    # 对每张图片进行提取关键点检测

    input = cv2.imread("D:\\test\\1\lfw\\11500.png")
    preds = fa.get_landmarks(input)[0]
    ans = visualize_facial_landmarks(input, preds, ["mouth"])
    # ans = process_image(preds, input)
    # 保存图片文件
    #cv2.imwrite("D:/test/1.ipg", ans)    #不行
    cv2.imshow("image", ans)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # im = Image.fromarray(input)
    # im.save("D:/test/10.jpg")



