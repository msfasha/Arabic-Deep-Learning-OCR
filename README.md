# ArabicOCR

The code in this model is based on the work of https://github.com/githubharald/SimpleHTR.

The details of this model was presented at:
https://arxiv.org/abs/2009.01987

Code to generate images can be found at:
https://github.com/JTCodeStore/TextIimagesToolkit

Samples of the used dataset (binary format) can be found at: https://drive.google.com/drive/folders/1mRefmN4Yzy60Uh7z3B6cllyyOXaxQrgg?usp=sharing



## Usage:

- Download code files from the repository (git clone git@github.com:msfasha/Arabic-Deep-Learning-OCR.git)
- Create python virtual environment in the created directory.
  sudo  apt-get install python3-venv
  python3 -m venv env
  
- Activate python virtual environment.
  source ./env/activate
  
- install ll the required python librararies (this script uses TensorFlow version 1.x).
	pip install -r requirements.txt

- Download sample dataset from:
  https://drive.google.com/drive/folders/1mRefmN4Yzy60Uh7z3B6cllyyOXaxQrgg?usp=sharing
  
  a suitable dataset for initial training is the (1_nice_60000_rows) dataset. 
  Download the two files of that dataset (1_nice_60000_rows.bin and 1_nice_60000_rows.txt).
- Save dataset files in the data folder of the project (the location of the data files can be changed in Config.py).
- Configurations of the run session can be adjusted in the Config.py files, this include folders names, the dataset files, the session type e.g. Train, Test, Infer.
- Initial setting of the configuration is:
  BASE_FILENAME = "1_nice_60000_rows"
  OPERATION_TYPE = OperationType.Training
  REGENERATE_CHARLIST_AND_CORPUS = True
  TRAINING_SAMPLES_PER_EPOCH = 5000
  BATCH_SIZE = 100
  IMAGE_WIDTH = 128
  IMAGE_HEIGHT = 32
  MAX_TEXT_LENGTH = 32

- In terminal window, goto src folder and run python Main.py.
- The code will generate training, validation and testing dataset from (1_nice_60000_rows) dataset.


