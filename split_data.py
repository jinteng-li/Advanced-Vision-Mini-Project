import os
import shutil

cur_dir = os.getcwd()
train_data_path = os.path.join(cur_dir, "data", "train")
val_data_path = os.path.join(cur_dir, "data", "val")
num_val = 10

train_list = os.listdir(train_data_path)

os.chdir(val_data_path)
for item in train_list:
    if item not in os.listdir(val_data_path):
        os.mkdir(item)

for dir in train_list:
    source_train_dir = os.path.join(train_data_path, dir)
    target_val_dir = os.path.join(val_data_path, dir)
    images = os.listdir(source_train_dir)
    if len(os.listdir(target_val_dir)) == num_val:
        continue

    for i in range(num_val):
        source = os.path.join(source_train_dir, images[i])
        target = os.path.join(target_val_dir, images[i])
        shutil.move(source, target)
