import os
from enum import Enum

# if os.name == 'nt':
#     BASE_PATH = 'd:/mywork/dev/ArabicOCR/'
# else:
#     BASE_PATH = '/mywork/dev/ArabicOCR/'

os.chdir('../')
BASE_PATH = os.getcwd()


class OperationType(Enum):
    Training = 1
    Validation = 2
    Testing = 3
    Infer = 4


class DecoderType:
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2


# Experiment name, to be saved in the audit log.
EXPERIMENT_NAME = "Test Drive Training Process"

# The name for the dataset files, the binary file and the labels file.
BASE_FILENAME = "1_nice_60000_rows"

# The type of the run session, depending on this type, designated datasets will be loaded.
OPERATION_TYPE = OperationType.Training

DECODER_TYPE = DecoderType.BestPath

# Use this value to regenerate the training/validation/test datasets, as well as
# the other support files. Usually this is needed when we start the training process
# It is not needed during the Testing process so we set it to true
# in order to regenerate all the required files, we have to delete to old ones train/validate/test and delete
# and then its value to true. After running the app, the files are generated and we can set it back to false unless
# we need to generate a new set of data i.e. train/validate/test
REGENERATE_CHARLIST_AND_CORPUS = True

DATA_PATH = BASE_PATH + '/dataset/'
MODEL_PATH = BASE_PATH + '/model/'
OUTPUT_PATH = BASE_PATH + '/output/'
INDIVIDUAL_TEST_IMAGE_PATH = DATA_PATH

BASE_IMAGES_FILE = DATA_PATH + BASE_FILENAME + ".bin"
BASE_LABELS_FILE = DATA_PATH + BASE_FILENAME + ".txt"
TRAINING_LABELS_FILE = DATA_PATH + "TRAINING_DATA_" + BASE_FILENAME + ".txt"
VALIDATION_LABELS_FILE = DATA_PATH + \
    "VALIDATION_DATA_" + BASE_FILENAME + ".txt"
TESTING_LABELS_FILE = DATA_PATH + "TESTING_DATA_" + BASE_FILENAME + ".txt"
fnCharList = OUTPUT_PATH + 'charList.txt'
fnResult = OUTPUT_PATH + 'result.txt'

# place whatever word image file you want to test/infer
fnInfer = INDIVIDUAL_TEST_IMAGE_PATH + "ae_Nice_7.png"

fnCorpus = OUTPUT_PATH + 'corpus.txt'
fnwordCharList = OUTPUT_PATH + 'wordCharList.txt'

# Number of batches for each epoch = SAMPLES_PER_EPOCH / BATCH_SIZE
TRAINING_SAMPLES_PER_EPOCH = 5000
BATCH_SIZE = 100
VALIDATIOIN_SAMPLES_PER_STEP = (int)(TRAINING_SAMPLES_PER_EPOCH * .2)
ACCUMULATED_PROCESSING_TIME = 0
TRAINING_DATASET_SIZE = .9
# .5 of the remaining ==> (Total - TRAINING_DATASET_SIZE) / 2
VALIDATION_DATASET_SPLIT_SIZE = .5
# stop after no improvements for this number of epochs
MAXIMUM_NONIMPROVED_EPOCHS = 5
MAXIMUM_MODELS_TO_KEEP = 3  # usually only 1, the last one

#IMAGE_SIZE = (128, 32)
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32
MAX_TEXT_LENGTH = 32
RESIZE_IMAGE = True
CONVERT_IMAGE_TO_MONOCHROME = False
MONOCHROME_BINARY_THRESHOLD = 127
AUGMENT_IMAGE = False


def auditLog(logStr):
    open(fnResult, 'a').write(logStr)
