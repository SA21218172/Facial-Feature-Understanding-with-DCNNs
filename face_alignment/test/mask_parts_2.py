import os
from folder_face_flip import mask, mask_box_add10
import cv2
import shutil
'''
与原始文件不同的是, 本文件可以mask掉对应文件夹下有很多class的图像文件
'''


def mk_file(file_path:str):
    if os.path.exists(file_path):
        return 0
        # rmtree(file_path)
    os.makedirs(file_path)
def save_err(image_list):
    if len(image_list) == 0:
        return
    err_path = "/home/qianqian/data0/err.txt"
    f = open(err_path, "w")
    for line in image_list:
        f.write(line + "\n")
    f.close()
    

def main():
    ori_path = "/home/qianqian/val_dataset/face_data"
    dst_path = "/home/qianqian/val_dataset/face_data_mouth_2"
    if os.path.exists(dst_path):
       pass
    else:
        mk_file(dst_path) 
    class_list = os.listdir(ori_path)
    err = 0
    mask_list = ["mouth"]
    temp = 0
    for cla in class_list:
        images_list = os.listdir(os.path.join(ori_path, cla))
        new_cla_path = os.path.join(dst_path, cla)
        mk_file(new_cla_path)
        for image in images_list:
            img_path = os.path.join(ori_path,cla, image)
            new_path = os.path.join(dst_path, cla, image)
            #new_path 转png
            # new_path = new_path[:-4] + ".png"

            try:
                mask_box_add10(img_path, mask_list, new_path, is_rand= False, is_the_same=True) #dataset里的比例有的不是很合适，会造成有的mask位置过大，有的mask不到的情况出现
                print("{} have been executed!".format(image))
            except TypeError:
                print(img_path, "Type error")
                # return
                shutil.copy(img_path, new_path)
                err += 1
            except ZeroDivisionError:
                print(img_path, "zero error")
                # return
                print("123")
                shutil.copy(img_path, new_path)
                err += 1
    print("error mount is {}".format(err))
        # save_err(err_list)

if __name__ == "__main__":
    main()

        


