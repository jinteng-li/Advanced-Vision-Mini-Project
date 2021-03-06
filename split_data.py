import os
import shutil

cur_dir = os.getcwd()
train_data_path = os.path.join(cur_dir, "data", "train")
val_data_path = os.path.join(cur_dir, "data", "val")

# num of images that will move to validation set(per class)
# decide train-val split ratio, since originally trainning set have 50 images per class
# ratio = (50-10) : 10 = 4:1
num_val = 10

train_list = os.listdir(train_data_path)

os.chdir(val_data_path)
# creating 1000 class folder in the validation directory
for item in train_list:
    if item not in os.listdir(val_data_path):
        os.mkdir(item)

# moving samples from training set folder to validation set folder
for dir in train_list:
    source_train_dir = os.path.join(train_data_path, dir)
    target_val_dir = os.path.join(val_data_path, dir)
    images = os.listdir(source_train_dir)
    
    # avoid multiple split of the training set
    if len(os.listdir(target_val_dir)) == num_val:
        continue

    for i in range(num_val):
        source = os.path.join(source_train_dir, images[i])
        target = os.path.join(target_val_dir, images[i])
        shutil.move(source, target)
