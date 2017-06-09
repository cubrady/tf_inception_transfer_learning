##############################################################
# Please config path FIRST
##############################################################
TF_PATH = "/home/brad_chang/proj/github/tensorflow/tensorflow"                  # The path of Tensorflow project
TRAINING_DATASET_PATH = "/data/deep_learning/dataset/training/pycon_2017_demo"  # The path of training dataset
TRAINED_MODEL_PATH = "/data/deep_learning/trainedModel/tf/pycon_2017_demo"      # The path to store retrained model
TESTING_DATASET_PATH = "/data/deep_learning/dataset/test/pycon_2017_demo"       # The path of testing dataset
MODEL_WORKSPACE = TRAINED_MODEL_PATH
##############################################################

TF_MODEL_NAME = "output_graph.pb"
TF_LABEL_NAME = "output_labels.txt"
LOG_FOLDER = "retrain_logs"


DIC_LABEL_FOLDER_MAP_V13 = {
    # folder    : label
    'unknown'   : "unknown",
    'fashion'   : 'fashion',
    'dog'       : 'dog',
    'hairstyle' : 'hairstyle',
    'food'      : "food",
    'babies'    : "babies",
    "bikini"    : 'bikini',
    'travel'    : "travel",
    'xxx'       : "xxx",
    "hotguys"   : 'hotguys',
    "nailart"   : 'nailart',
    'text'      : "text",
    'flower'    : "flower",
    'wedding'   : "wedding",
    'art'       : "art",
    'xbutt'     : "xbutt",
    'cat'       : "cat",
    'xboobs'    : "xboobs",
    'animation' : "animation",
    }


DIC_LABEL_VER_MAP = {
    13 : DIC_LABEL_FOLDER_MAP_V13
}
