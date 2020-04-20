#filenames and paths to data
import os
from enum import Enum
from shutil import copyfile
if os.name == 'nt':
    BASE_PATH = 'D:/Mywork/HTR/'
else:
    BASE_PATH = '/content/drive/My Drive/MyWork/HTRDataFiles/'
    #BASE_PATH = '/content/sample_data/HTRDataFiles/'


class OperationType(Enum):
    Training = 1
    Validation = 2
    Testing = 3
    Infer = 4

class DecoderType:
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2

EXPERIMENT_NAME = "Testing conv dumping"
BASE_FILENAME = "18_fonts_450000_unique_word_samples"
OPERATION_TYPE = OperationType.Infer
DECODER_TYPE = DecoderType.BestPath

#the below value only needs to be true during training
#for testing, we should use the old/original value of the model
REGENERATE_CHARLIST_AND_CORPUS = False

DATA_PATH = BASE_PATH  + 'Data/'
DATASET_PATH = DATA_PATH  + 'CurrentExperimentData/'
OUTPUT_PATH = DATA_PATH + 'CurrentExperimentOutput/'
MODEL_PATH = DATA_PATH + 'CurrentExperimentModel/'
INDIVIDUAL_TEST_IMAGE_PATH = DATA_PATH + 'IndividualTestImages/'

BASE_IMAGES_FILE = DATASET_PATH + BASE_FILENAME + ".bin"
BASE_LABELS_FILE = DATASET_PATH + BASE_FILENAME + ".txt"
TRAINING_LABELS_FILE = DATASET_PATH + "TRAINING_DATA_" + BASE_FILENAME + ".txt"
VALIDATION_LABELS_FILE = DATASET_PATH  + "VALIDATION_DATA_" + BASE_FILENAME + ".txt"
TESTING_LABELS_FILE = DATASET_PATH  + "TESTING_DATA_" + BASE_FILENAME + ".txt"
fnCharList = OUTPUT_PATH + 'charList.txt'
fnResult = OUTPUT_PATH + 'result.txt'
fnInfer = INDIVIDUAL_TEST_IMAGE_PATH + "ae_Nice_7.png"
fnCorpus = OUTPUT_PATH + 'corpus.txt'
fnwordCharList = OUTPUT_PATH + 'wordCharList.txt'

#Number of batches for each epoch = SAMPLES_PER_EPOCH / BATCH_SIZE
TRAINING_SAMPLES_PER_EPOCH = 25000
BATCH_SIZE = 50
VALIDATIOIN_SAMPLES_PER_STEP = (int)(TRAINING_SAMPLES_PER_EPOCH * .2)
ACCUMULATED_PROCESSING_TIME = 0
TRAINING_DATASET_SIZE = .9
VALIDATION_DATASET_SPLIT_SIZE = .5 #.5 of the remaining ==> (Total - TRAINING_DATASET_SIZE) / 2
MAXIMUM_NONIMPROVED_EPOCHS = 5 #stop after no improvements for this number of epochs
MAXIMUM_MODELS_TO_KEEP = 3 #usually only 1, the last one

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
