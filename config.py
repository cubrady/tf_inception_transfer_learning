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
TEST_RESULT_FILE = "test_result.log"

DIC_LABEL_FOLDER_MAP_V13 = {
    # folder    : label
    'unknown'   : "unknown",
    'fashion'   : 'fashion',
    'food'      : "food",
    'travel'    : "travel",
    'xxx'       : "xxx",
    }


DIC_LABEL_VER_MAP = {
    5 : DIC_LABEL_FOLDER_MAP_V13
}
