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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    model = iresnet100(pretrained=True)
    target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    ori_path = "D:/face_data/val_bn"
    dst_final_path = "D:/result_grad_cam/face_data_val_bn"
    mk_dir(dst_final_path)
    folder_list = os.listdir(ori_path)
    for folder in folder_list:
        img_folder_path = os.path.join(ori_path, folder)
        dst_folder_path = os.path.join(dst_final_path, folder)
        mk_dir(dst_folder_path)

        for img in os.listdir(img_folder_path):
            img_path = os.path.join(img_folder_path, img)
            dst_path = os.path.join(dst_folder_path, img)
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img, dtype=np.uint8)
            # img = center_crop_img(img, 224)

            # [C, H, W]
            img_tensor = data_transform(img)
            # expand batch dimension
            # [C, H, W] -> [N, C, H, W]
            input_tensor = torch.unsqueeze(img_tensor, dim=0)

            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
            target_category = int(folder)  # tabby, tabby cat
            # target_category = 254  # pug, pug-dog

            grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                              grayscale_cam,
                                              use_rgb=False)

            outputs = model(input_tensor.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            predict_y = int(predict_y)
            if predict_y == target_category:
                predict_y = 0
            else:
                predict_y = 1
            dst_path = dst_path[:-4] + "_" + str(predict_y) + ".jpg"
            # im = Image.fromarray(visualization).convert('RGB')
            # im.save(dst_path, quality=95)
            # cv2.imwrite(dst_path, visualization)
            plt.imshow(visualization)
            plt.show()


if __name__ == '__main__':
    main()
