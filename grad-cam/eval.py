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

    image_path = "D:/face_data"
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)


    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    test_list = ["val"]
    # test_list = ["val", "val_eyes", "val_mouth", "val_nose", "val_em", "val_en", "val_mn"]
    # test_list = ["val_em", "val_en", "val_mn"]
    for part in test_list:
        temp_path = os.path.join(image_path, part)
        validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, part),
                                                transform=data_transform["val"])
        val_num = len(validate_dataset)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                      batch_size=batch_size, shuffle=False,
                                                      num_workers=nw)

        print(" {} images for validation.".format(val_num))

        net = iresnet100(pretrained=True)

        #
        # for param in net.parameters():
        #     param.requires_grad = False
        net.to(device)
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
        print('%s,val_accuracy: %.3f' % (part, val_accurate))

if __name__ == '__main__':
    main()
