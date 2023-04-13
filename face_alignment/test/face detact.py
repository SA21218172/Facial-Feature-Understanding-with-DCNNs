import face_alignment
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

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

#对每张图片进行提取关键点检测
for i in range(1,6):
    input = io.imread("../test/assets/{}.jpg".format(i))
    preds = fa.get_landmarks(input)
    # plt.imshow(input)
    # for detection in preds:
    #     plt.scatter(detection[:, 0], detection[:, 1], 2)
    #     for idx, point in enumerate(detection):
    #         pos = (int(point[0]), int(point[1]))
    #         print(idx, pos)
    #         # 利用cv2.putText输出1-68
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         cv2.putText(input, str(idx + 1), pos, font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    #         plt.imshow(input)
    #
    # #io.imshow(preds)
    # plt.show()

    #处理左眼
    array_left_eye = preds[0][36:42]
    left_eye_box = get_bounding_box(array_left_eye)

    #处理右眼
    array_right_eye = preds[0][42:48]
    right_eye_box = get_bounding_box(array_right_eye)

    #处理嘴巴

    #图片位置翻转
    ans = filp_part(input, left_eye_box)
    #下面的语句将input进行显示与ans的矩阵相同，涉及到浅copy
    # plt.imshow(input)
    plt.imshow(ans)
    plt.show()





    print("here")

