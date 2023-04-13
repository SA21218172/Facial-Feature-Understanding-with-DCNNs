import os
from folder_face_flip import mask, mask_box_add10
import cv2
import shutil
# mask(image_path="/home/qianqian/extract_rec_file/00000.png", mask_parts=["mouth"], save_path="/home/qianqian/extract_rec_file/0.png")


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
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    ori_path = "/home/qianqian/val_dataset/origin_data/cfp_fp"
    dst_path = "/home/qianqian/val_dataset/mask_box_data/cfp_fp_mn"
    if os.path.exists(dst_path):
       pass
    else:
        mk_file(dst_path) 
    err = 0
    err_list = []
    images_list = os.listdir(ori_path)
    num = len(images_list)
    print("the count of images is {}".format(num))
    mask_list = ["nose", "mouth"]
    temp = 0
    for image in images_list:
        img_path = os.path.join(ori_path,image)
        new_path = os.path.join(dst_path, image)
        try:
            mask_box_add10(img_path, mask_list, new_path,is_rand= False, is_the_same=True) #dataset里的比例有的不是很合适，会造成有的mask位置过大，有的mask不到的情况出现
            print("{} have been executed!".format(image))
        except TypeError:
            print(img_path, "Type error")
            shutil.copy(img_path, new_path)
            err_list.append(image)
            err += 1
        except ZeroDivisionError:
            print(img_path, "Type error")
            shutil.copy(img_path, new_path)
            err_list.append(image)
            err += 1
    print("error mount is {}".format(err))
    print(err_list)
    # save_err(err_list)

if __name__ == "__main__":
    main()

        


