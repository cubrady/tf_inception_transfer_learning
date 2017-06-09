# -*- coding: utf8 -*-

import os
import time
import argparse
from config import TRAINING_DATASET_PATH, TF_MODEL_NAME, TF_LABEL_NAME, TF_PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow model training helper')
    parser.add_argument('version')
    args = parser.parse_args()

    currentVer = args.version
    trainedModelPath = os.path.join(TRAINING_DATASET_PATH, "v%s" % currentVer)
    os.makedirs(trainedModelPath)

    start = time.time()

    cmdTrain = "cd %s && bazel-bin/tensorflow/examples/image_retraining/retrain —print_misclassified_test_images --image_dir %s" % (TF_PATH, TRAINING_DATASET_PATH)
    cmdCopyModel = "cp /tmp/%s %s" % (TF_MODEL_NAME, os.path.join(trainedModelPath, TF_MODEL_NAME))
    cmdCopyModelLabel = "cp /tmp/%s %s" % (TF_LABEL_NAME, os.path.join(trainedModelPath, TF_LABEL_NAME))
    cmdCopyModelLog = "cp -r /tmp/retrain_logs %s" % (trainedModelPath)
    cmdTBoard = "tensorboard --logdir /tmp/retrain_logs"
    lstCmds = [cmdTrain, cmdCopyModel, cmdCopyModelLabel, cmdCopyModelLog, cmdTBoard]

    for cmd in lstCmds:
        os.system(cmd)

    print "Spend %f sec" % (time.time() - start)
