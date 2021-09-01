from __future__ import division
from __future__ import print_function

import os
import random
from shutil import Error
import sys
import Config as config
import numpy as np
from SamplePreprocessor import preprocess


class Sample:
    "a single sample from the dataset"

    def __init__(self, gtText, imageIdx, imageHeight, imageWidth, imageSize, imageStartPosition):
        self.gtText = gtText
        self.imageIdx = imageIdx
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        self.imageSize = imageSize
        self.imageStartPosition = imageStartPosition


class Batch:
    "batch containing images and ground truth texts"

    def __init__(self, gtTexts, imgs):
        self.gtTexts = gtTexts
        self.imgs = np.stack(imgs, axis=0)


class DataGenerator:

    def __init__(self):
        self.binaryImageFile = open(config.BASE_IMAGES_FILE, "rb")
        self.currIdx = 0
        self.samples = []
        self.trainSamples = []
        self.validationSamples = []
        self.testSamples = []

    def LoadData(self, operationType):
        if not os.path.isfile(config.TRAINING_LABELS_FILE) \
                or not os.path.isfile(config.VALIDATION_LABELS_FILE) \
                or not os.path.isfile(config.TESTING_LABELS_FILE):
            self.createDataFiles()

        if operationType == config.OperationType.Training:
            self.loadDataFile(config.OperationType.Training)
            self.loadDataFile(config.OperationType.Validation)
        elif operationType == config.OperationType.Validation:
            self.loadDataFile(config.OperationType.Validation)
        elif operationType == config.OperationType.Testing:
            self.loadDataFile(config.OperationType.Testing)

    def createDataFiles(self):
        charsSet = set()
        wordsSet = set()

        f = open(config.BASE_LABELS_FILE, encoding="utf-8")
        for line in f:
            # read all samples ==> append line as is
            self.samples.append(line)

            if config.REGENERATE_CHARLIST_AND_CORPUS:
                # extract unique characters from text
                lineSplit = line.split(';')
                gtText = lineSplit[8]
                gtText = gtText[5:]
                wordsSet.add(gtText)
                charsSet = charsSet.union(set(list(gtText)))

        f.close()

        # create a text file that contains all the characters in the dataset
        # this list shall used to create the CTC model
        # There might be a problem if a previously saved model used larger data, consequently, not all
        # the characters in the previous model will be generated and therefore RNN creation will fail
        # note that a problem might arise when we try to open a saved model that was saved on a larger dataset
        # conseuqnelty some represented characters might be abscent and the new model will fail to load previous one
        # a solution for this problem is to use a static character set for the used dataset
        # also create the corpus data file for BeamSearch (if required)

        # DONT CREATE THEM UNLESS U R USING LARGER DATASET, ALREADY CREATED IN DIRECTORY
        if config.REGENERATE_CHARLIST_AND_CORPUS:
            localCharList = sorted(list(charsSet))
            open(config.fnCharList, 'w',
                 encoding="utf-8").write(str().join(localCharList))
            open(config.fnCorpus, 'w',
                 encoding="utf-8").write(str().join(sorted(list(wordsSet))))

        # first of all, make sure to randomly shuffle the main lables file
        # random.shuffle(self.samples)

        # split into training, validation, testing
        lenOfAllSamples = len(self.samples)
        lenOfTrainSamples = int(config.TRAINING_DATASET_SIZE * lenOfAllSamples)
        lenOfTrainingAndValidationSamples = lenOfAllSamples - lenOfTrainSamples
        lenOfValidationSamples = int(
            config.VALIDATION_DATASET_SPLIT_SIZE * lenOfTrainingAndValidationSamples)

        with open(config.TRAINING_LABELS_FILE, 'w', encoding="utf-8") as f:
            for item in self.samples[:lenOfTrainSamples]:
                f.write(item)

        with open(config.VALIDATION_LABELS_FILE, 'w', encoding="utf-8") as f:
            for item in self.samples[lenOfTrainSamples:lenOfTrainSamples + lenOfValidationSamples]:
                f.write(item)

        with open(config.TESTING_LABELS_FILE, 'w', encoding="utf-8") as f:
            for item in self.samples[lenOfTrainSamples + lenOfValidationSamples:]:
                f.write(item)

        self.samples = []

    def loadDataFile(self, operationType):
        if operationType == config.OperationType.Training:
            fileName = config.TRAINING_LABELS_FILE
        elif operationType == config.OperationType.Validation:
            fileName = config.VALIDATION_LABELS_FILE
        elif operationType == config.OperationType.Testing:
            fileName = config.TESTING_LABELS_FILE

        f = open(fileName, encoding="utf-8")
        for line in f:
            lineSplit = line.split(';')

            imgIdx = lineSplit[0]
            imgIdx = imgIdx[10:]

            imgStartPosition = lineSplit[1]
            imgStartPosition = int(imgStartPosition[15:])

            imgHeight = lineSplit[2]
            imgHeight = int(imgHeight[13:])
            imgWidth = lineSplit[3]
            imgWidth = int(imgWidth[12:])
            imgSize = imgHeight * imgWidth

            gtText = lineSplit[8]
            gtText = gtText[5:]
            #gtText = self.truncateLabel(' '.join(gtText), config.MAX_TEXT_LENGTH)

            # put sample into list
            if operationType == config.OperationType.Training:
                self.trainSamples.append(
                    Sample(gtText, imgIdx, imgHeight, imgWidth, imgSize, imgStartPosition))
            elif operationType == config.OperationType.Validation:
                self.validationSamples.append(
                    Sample(gtText, imgIdx, imgHeight, imgWidth, imgSize, imgStartPosition))
            elif operationType == config.OperationType.Testing:
                self.testSamples.append(
                    Sample(gtText, imgIdx, imgHeight, imgWidth, imgSize, imgStartPosition))

        # if operationType == config.OperationType.Training:
        #     self.trainWords = [x.gtText for x in self.trainSamples]
        # elif operationType == config.OperationType.Validation:
        #     self.validationWords = [x.gtText for x in self.validationSamples]
        # elif operationType == config.OperationType.Testing:
        #     self.testWords = [x.gtText for x in self.testSamples]

    def truncateLabel(self, text, maxTextLen):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]
        return text

    def selectTrainingSet(self):
        "switch to randomly chosen subset of training set"
        self.currIdx = 0
        random.shuffle(self.trainSamples)
        self.samples = self.trainSamples[:config.TRAINING_SAMPLES_PER_EPOCH]

    def selectValidationSet(self):
        "switch to validation set"
        self.currIdx = 0
        random.shuffle(self.validationSamples)
        self.samples = self.validationSamples[:
                                              config.VALIDATIOIN_SAMPLES_PER_STEP]

    def selectTestSet(self):
        "switch to validation set"
        self.currIdx = 0
        random.shuffle(self.testSamples)
        self.samples = self.testSamples[:config.VALIDATIOIN_SAMPLES_PER_STEP]

    def getIteratorInfo(self):
        "current batch index and overall number of batches"
        return (self.currIdx // config.BATCH_SIZE + 1, len(self.samples) // config.BATCH_SIZE)

    def hasNext(self):
        "iterator"
        return self.currIdx + config.BATCH_SIZE <= len(self.samples)

    def getNext(self):
        "iterator"
        batchRange = range(self.currIdx, self.currIdx + config.BATCH_SIZE)
        gtTexts = [self.samples[i].gtText for i in batchRange]

        imgs = []
        for i in batchRange:
            try:
                self.binaryImageFile.seek(self.samples[i].imageStartPosition)
                img = np.frombuffer(self.binaryImageFile.read(
                    self.samples[i].imageSize), np.dtype('B'))
                img = img.reshape(
                    self.samples[i].imageHeight, self.samples[i].imageWidth)
                img = preprocess(img)
                # img = preprocess(img, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.RESIZE_IMAGE,
                #                  config.CONVERT_IMAGE_TO_MONOCHROME, config.AUGMENT_IMAGE)
                imgs.append(img)
            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
                pass
            except ValueError as e:
                print("Value error({0}): {1}".format(e.errno, e.strerror))
                pass
            except Error as e:
                print("Unexpected error:", sys.exc_info()[0])
                print("Value error({0}): {1}".format(e.errno, e.strerror))

                pass

        self.currIdx += config.BATCH_SIZE
        return Batch(gtTexts, imgs)
