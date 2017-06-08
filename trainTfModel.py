# -*- coding: utf8 -*-

import os
import argparse

TF_PATH = "/home/brad_chang/proj/github/tensorflow/tensorflow"
TRAINING_SRC = "/data/deep_learning/dataset/training/pg_common_labels/"
TRAINED_MODEL_PATH = "/data/deep_learning/trainedModel/tf/pg_label_set"

TF_MODEL_NAME = "output_graph.pb"
TF_LABEL_NAME = "output_labels.txt"
LOG_FOLDER = "retrain_logs"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow model training helper')
    parser.add_argument('version')
    args = parser.parse_args()

    currentVer = args.version
    trainedModelPath = os.path.join(TRAINED_MODEL_PATH, "v%s" % currentVer)
    os.makedirs(trainedModelPath)

    cmdTrain = "cd %s && bazel-bin/tensorflow/examples/image_retraining/retrain —print_misclassified_test_images --image_dir %s" % (TF_PATH, TRAINING_SRC)
    cmdCopyModel = "cp /tmp/%s %s" % (TF_MODEL_NAME, os.path.join(trainedModelPath, TF_MODEL_NAME))
    cmdCopyModelLabel = "cp /tmp/%s %s" % (TF_LABEL_NAME, os.path.join(trainedModelPath, TF_LABEL_NAME))
    cmdCopyModelLog = "cp -r /tmp/retrain_logs %s" % (trainedModelPath)
    cmdTBoard = "tensorboard --logdir /tmp/retrain_logs"
    lstCmds = [cmdTrain, cmdCopyModel, cmdCopyModelLabel, cmdCopyModelLog, cmdTBoard]

    for cmd in lstCmds:
        os.system(cmd)
