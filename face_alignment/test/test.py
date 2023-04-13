import os


path = "D:/mslm_data"
path_2 = "D:/mslm_data_flip/"
num_origin = len(os.listdir(path))
num_new = len(os.listdir(path_2))

sum = 0
for i in os.listdir(path):
    path_img = os.path.join(path, i)
    sum += len(os.listdir(path_img))

print(sum)