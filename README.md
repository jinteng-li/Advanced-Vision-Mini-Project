# Advanced-Vision-Mini-Project
This project is built to form our solution to the VIPriors image classification challenge
Our code is developed based on the image classification toolkit provided by VIPriors: https://github.com/VIPriors/vipriors-challenges-toolkit/tree/master/image-classification
We modified the following files:
1. main.py, (1)we add extra arguments for turning on and off data augmentation strategy and altering loss function.
            (2)we also made a minor change to save the submission csv file, instead of storing two columns, we change the code to only store one column of predictions                  for each test sample
            (3)We add code in main.py to apply data augumentation strategies including Mixup, Cutmix and AutoAugment 
2. evaluate_classification.py, due to the submission csv file format change, we made corresponding changes in evaluate_classification.py to make the evalution work as
                               usually.

Files that we create(not provided by the VIPrior toolkit)
1. loss.py, covers all the alternation loss function that we use in our project: JSD, label smoothing and constructive learning
2. split_data.py, helper script that split a portion of training set as validation set(train:valid ratio=4:1)

All the other files are provided by VIPriors toolkit and remain unchanged.

Note: we made a small modification(adding one extra 'Prediction' line at the beginning) to the ground truth file in order to coordinate with the csv file.
      the modified ground_truth file is uploaded in this repository.
