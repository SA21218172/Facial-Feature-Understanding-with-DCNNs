import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from iresnet import iresnet100
import cv2
def mk_dir(path: str):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def extract_grayscale_cam(indices_key, indices_value):
    # key是类别/名称， value是对应每个feature的位置索引的字典
    img_indices = indices_key

    # get the loc of the features, 对应顺序：[left, right, up, down]
    l_e = indices_value['eyes'][:4]
    r_e = indices_value['eyes'][4:]
    l_b = indices_value['eyebrow'][:4]
    r_b = indices_value['eyebrow'][4:]
    m = indices_value['mouth']
    n = indices_value['nose']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = iresnet100(pretrained=True)
    target_layers = [model.layer4]


    data_transform = transforms.Compose([transforms.ToTensor(),
                                         # transforms.Resize((112,112)),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    ori_path = "/home/qianqian/dataset/face_data_2_val_eyebrow"
    # print("ori_path is {}".format(ori_path))
    img_path = os.path.join(ori_path, img_indices)
    img_label = img_indices[:5]
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 112)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = int(img_label)  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :] * 255.0

    avg_le = com_avg_value_of_mask(grayscale_cam, l_e)
    avg_rl = com_avg_value_of_mask(grayscale_cam, r_e)
    avg_eye = (avg_rl + avg_le) / 2.0
    avg_lb = com_avg_value_of_mask(grayscale_cam, l_b)
    avg_rb = com_avg_value_of_mask(grayscale_cam, r_b)
    avg_eyebrow = (avg_rb + avg_lb) / 2.0
    avg_mouth = com_avg_value_of_mask(grayscale_cam, m)
    avg_nose = com_avg_value_of_mask(grayscale_cam, n)
    return [avg_eye, avg_eyebrow, avg_mouth, avg_nose]
    # 可视化灰度cam
    # unloader = transforms.ToPILImage()
    # temp = unloader(grayscale_cam)
    # temp = np.array(temp, dtype=np.float32)
    # temp = temp.astype(dtype=np.float32)
    # plt.imshow(temp)
    # plt.show()


def com_avg_value_of_mask(cam, loc_of_feature):
    '''
    @param cam: shape: h*w
    @param loc_of_feature:
    @return: 对应位置的平均值
    '''

    left, right, up, down = loc_of_feature
    region = cam[up:down, left:right]
    avg = np.mean(region)
    return avg



if __name__ == '__main__':
    print("success")
