"""
Preparing model:
 - Install bazel ( check tensorflow's github for more info )
    Ubuntu 14.04:
        - Requirements:
            sudo add-apt-repository ppa:webupd8team/java
            sudo apt-get update
            sudo apt-get install oracle-java8-installer
        - Download bazel, ( https://github.com/bazelbuild/bazel/releases )
          tested on: https://github.com/bazelbuild/bazel/releases/download/0.2.0/bazel-0.2.0-jdk7-installer-linux-x86_64.sh
        - chmod +x PATH_TO_INSTALL.SH
        - ./PATH_TO_INSTALL.SH --user
        - Place bazel onto path ( exact path to store shown in the output)
- For retraining, prepare folder structure as
    - root_folder_name
        - class 1
            - file1
            - file2
        - class 2
            - file1
            - file2
- Clone tensorflow
- Go to root of tensorflow
- bazel build tensorflow/examples/image_retraining:retrain
- bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir /path/to/root_folder_name  --output_graph /path/output_graph.pb --output_labels /path/output_labels.txt --bottleneck_dir /path/bottleneck
** Training done. **
For testing through bazel,
    bazel build tensorflow/examples/label_image:label_image && \
    bazel-bin/tensorflow/examples/label_image/label_image \
    --graph=/path/output_graph.pb --labels=/path/output_labels.txt \
    --output_layer=final_result \
    --image=/path/to/test/image
For testing through python, change and run this code.
"""

# Retrain
# bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir /home/brad_chang/deep_learning/dataset/xxx_detection/train
#
# cp /tmp/output_graph.pb /home/brad_chang/deep_learning/trainedModel/tf/xxx_recog/v5/output_graph.pb && cp /tmp/output_labels.txt /home/brad_chang/deep_learning/trainedModel/tf/xxx_recog/v5/output_labels.txt

import time
import os
import numpy as np
import tensorflow as tf
from config import *

TF_SESSION = None
SOFTMAX_TENSOR = None

def initTfModel(worksapce):
    t = time.time()
    create_graph(os.path.join(worksapce, TF_GRAPH_MODEL_NAME))
    label_lines = load_labels(os.path.join(worksapce, TF_GRAPH_LABEL_NAME))
    print "Model load complete, spend: %.3f sec" % (time.time() - t)
    return label_lines

def create_graph(graphModelPath):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(graphModelPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    global TF_SESSION
    global SOFTMAX_TENSOR
    TF_SESSION = tf.Session()
    SOFTMAX_TENSOR = TF_SESSION.graph.get_tensor_by_name('final_result:0')

def load_labels(path):
    label_lines = []
    for l in open(path, "r"):
        label_lines.append(l.replace("\n", ""))
    return label_lines

def analyzeIamge(image_path, label_lines):
    ret = []
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
        return ret

    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    try:
        predictions = TF_SESSION.run(SOFTMAX_TENSOR, {'DecodeJpeg/contents:0': image_data})
    except Exception as err:
        print "Err: %s" % image_path
        print err
        return ret

    predictions = np.squeeze(predictions)

    top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
    labels = label_lines#[str(w).replace("\n", "") for w in label_lines]
    for node_id in top_k:
        human_string = labels[node_id]
        score = predictions[node_id]
        #print('%s (score = %.5f)' % (human_string, score))
        ret.append((human_string, score))

    return ret#answer, score, labels[top_k[1]], predictions[top_k[1]]

if __name__ == '__main__':
    #validateImages()
    #testImages()
    # bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir ~/flower_photos
    pass
