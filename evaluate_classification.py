"""
Use this script to evaluate your model. It stores metrics in the file
`scores.txt`.
Input:
    predictions (str): filepath. Should be a file that matches the submission
        format;
    groundtruths (str): filepath. Should be an annotation file.
Usage:
    evaluate_classification.py <groundtruths> <predictions> <output_dir>
"""

import numpy as np
import pandas as pd
import os
import sys

OUTPUT_FILE = 'scores.txt'

def evaluate_from_files(groundtruths_filepath, predictions_filepath, output_dir):

    output_dir = output_dir
    data = pd.read_csv(groundtruths_filepath)
    sub_data = pd.read_csv(predictions_filepath)

    ground_truth = data.to_numpy()
    submission = sub_data.to_numpy()


    indexed_gt = []
    for idx in range(len(ground_truth)):
        indexed_gt.append(ground_truth[idx][0])

    indexed_sbm = []
    for idx in range(len(submission)):
        indexed_sbm.append(submission[idx][0])


    tp = 0.0
    fp = 0.0
    for i in range(len(indexed_gt)):
        if indexed_gt[i] == indexed_sbm[i]:
            tp += 1.
        else:
            fp += 1.
    acc = tp / (tp+fp)

    print('accuracy', acc)

    metrics = [("Top1 accuracy", acc)]
    with open(os.path.join(output_dir, OUTPUT_FILE), 'w') as f:
            for name, val in metrics:
                f.write(f"{name}: {val:.8f}\n")

    print("Metrics written to scores.txt.")

if __name__ == '__main__':
    args = sys.argv[1:]
    evaluate_from_files(args[0], args[1], args[2])