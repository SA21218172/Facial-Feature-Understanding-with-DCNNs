import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from torchsummary import summary
from iresnet import iresnet100


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # data_transform = {"val": transforms.Compose([transforms.ToTensor(),
    #                                              transforms.Normalize([0.500, 0.500, 0.500], [0.500, 0.500, 0.500])])}
    image_path = "D:/face_data"
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)


    batch_size = 1
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # test_list = ["val", "val_eyes", "val_mouth", "val_nose", "val_eyebrow", "val_be", "val_bm", "val_bn", "val_em", "val_en", "val_mn"]
    # test_list = ["val", "val_eyes", "val_mouth", "val_nose", "val_em", "val_en", "val_mn"]
    # test_list = ["val_em", "val_en", "val_mn"]
    # test_list = ["val_mouth_3", "val_nose_3", "val_eyes_3", "val_eyebrow_3"]
    # test_list = ["val_rand_black"]
    # , "val_eyebrow_white", "val_eyes_white", "val_mouth_white", "val_nose_white"]
    # ]
    # test_list = ["val_eyes_rank_4", "val_mouth_rank_4", "val_nose_rank_4", "val_eyebrow_rank_4"]
    test_list = ["val_high_pass"]
    for part in test_list:
        temp_path = os.path.join(image_path, part)
        validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, part),
                                                transform=data_transform["val"])
        val_num = len(validate_dataset)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                      batch_size=batch_size, shuffle=False,
                                                      num_workers=nw)

        print(" {} images for validation.".format(val_num))

        net = iresnet100()
        net.features = nn.Sequential(nn.Linear(512, 512))
        net.to(device)

        # summary(net, (3, 112, 112))

        # load pretrain weights
        model_weight_path = "./iresnet100_face_2_classification.pth"
        net.load_state_dict(torch.load(model_weight_path, map_location=device))
        for param in net.parameters():
            param.requires_grad = False

        # validate
        #写个可以一次性测三个的
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('%s,val_accuracy: %.5f' % (part, val_accurate))

if __name__ == '__main__':
    main()

