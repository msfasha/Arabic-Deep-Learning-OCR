from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import time
import cv2
import editdistance
from DataGenerator_BinaryFile import DataGenerator, Batch
from Model import Model
from SamplePreprocessor import preprocess
import Config as config

startTime = datetime.now()

# we only need DataGenerator in training, validation, testing inorder to access the related datasets
if config.OPERATION_TYPE != config.OperationType.Infer:
    dataGenerator = DataGenerator()

def train(paraModel):
    "train NN"
    epoch = 0  # number of training epochs since start
    bestCharErrorRate = float('inf')  # best valdiation character error rate
    noImprovementSince = 0  # number of epochs no improvement of character error rate occured

    auditString = get_initial_status_log()
    print(auditString)
    config.auditLog(auditString)

    continueLooping = True

    while continueLooping:
        print("Current Time =", datetime.now())

        epoch += 1
        print('Epoch:', epoch)

        dataGenerator.selectTrainingSet()

        while dataGenerator.hasNext():

            timeSnapshot = time.time()

            iterInfo = dataGenerator.getIteratorInfo()
            batch = dataGenerator.getNext()
            loss = paraModel.trainBatch(batch)

            # #stop execution after reaching a certain threashold
            # if (int(loss) == 1):
            #     noImprovementSince = config.MAXIMUM_NONIMPROVED_EPOCHS;

            print('Training Batch:', iterInfo[0],
                  '/', iterInfo[1], 'Loss:', loss)

            accumulateProcessingTime(timeSnapshot)

        # validate
        charErrorRate, charSuccessRate, wordsSuccessRate = validate(
            paraModel, config.OperationType.Validation)
        auditString = "Epoch Number %d." % epoch + "\n"

        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            auditString = auditString + 'Character error rate improved, saving model'
            paraModel.save()
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
        else:
            auditString = auditString + "Character error rate not improved\n"
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= config.MAXIMUM_NONIMPROVED_EPOCHS:
            auditString = auditString + \
                "No more improvement since %d epochs." % config.MAXIMUM_NONIMPROVED_EPOCHS + "\n"

            # gracefull termination
            continueLooping = False

        # Model did not finish, print log and save it
        auditString = auditString + \
            get_execution_log(charSuccessRate, wordsSuccessRate)
        print(auditString)
        config.auditLog(auditString)


def validate(paraModel, paraOperationType):
    if paraOperationType == config.OperationType.Validation:
        dataGenerator.selectValidationSet()

    elif paraOperationType == config.OperationType.Testing:
        dataGenerator.selectTestSet()

    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    timeSnapshot = 0.0

    while dataGenerator.hasNext():
        timeSnapshot = time.time()

        iterInfo = dataGenerator.getIteratorInfo()
        print('Validating Batch:', iterInfo[0], '/', iterInfo[1])
        batch = dataGenerator.getNext()
        (recognized, _) = paraModel.inferBatch(batch)

        accumulateProcessingTime(timeSnapshot)

        # print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordTotal += 1
            numCharTotal += len(batch.gtTexts[i])

            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0

            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist

            # remove remark to see each success and error values
            #print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')

    # print validation result
    charErrorRate = numCharErr / numCharTotal
    charSuccessRate = 1 - (numCharErr / numCharTotal)
    wordsSuccessRate = numWordOK / numWordTotal

    # print and save validation result, this includes post epoch operation as well as when
    # running standalone testing or validation processes

    return charErrorRate, charSuccessRate, wordsSuccessRate


def inferSingleImage(paraModel, paraFnImg):
    "recognize text in image provided by file path"
    img = cv2.imread(paraFnImg, cv2.IMREAD_GRAYSCALE)
    img = preprocess(img, config.IMAGE_WIDTH,
                     config.IMAGE_HEIGHT, True, False, False)

    batch = Batch(None, [img])
    #(recognized, probability) = model.inferBatch(batch)
    (recognized, probability) = paraModel.inferBatch(batch, True)
    print('Recognized:', '"' + recognized[0] + '"')
    print('Probability:', probability[0])


def get_initial_status_log():
    auditString = "____________________________________________________________" + "\n"
    auditString = auditString + "Experiment Name: " + config.EXPERIMENT_NAME + "\n"
    auditString = auditString + "Base File Name: " + config.BASE_FILENAME + "\n"
    auditString = auditString + 'Start Execution Time :' + \
        startTime.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
    auditString = auditString + "Training set size: " + \
        str(len(dataGenerator.trainSamples)) + "\n"
    auditString = auditString + "Validation set size: " + \
        str(len(dataGenerator.validationSamples)) + "\n"
    auditString = auditString + "Training Samples per epoch: " + \
        str(config.TRAINING_SAMPLES_PER_EPOCH) + "\n"
    auditString = auditString + "Validation Samples per step: " + \
        str(config.VALIDATIOIN_SAMPLES_PER_STEP) + "\n"
    auditString = auditString + "Batch size: " + str(config.BATCH_SIZE) + "\n"
    auditString = auditString + "TRAINING_SAMPLES_PER_EPOCH: " + \
        str(config.TRAINING_SAMPLES_PER_EPOCH) + "\n"
    auditString = auditString + "BATCH_SIZE: " + str(config.BATCH_SIZE) + "\n"
    auditString = auditString + "VALIDATIOIN_SAMPLES_PER_STEP: " + \
        str(config.VALIDATIOIN_SAMPLES_PER_STEP) + "\n"
    auditString = auditString + "TRAINING_DATASET_SIZE: " + \
        str(config.TRAINING_DATASET_SIZE) + "\n"
    auditString = auditString + "VALIDATION_DATASET_SPLIT_SIZE: " + \
        str(config.VALIDATION_DATASET_SPLIT_SIZE) + "\n"
    auditString = auditString + "IMAGE_WIDTH: " + \
        str(config.IMAGE_WIDTH) + "\n"
    auditString = auditString + "IMAGE_HEIGHT: " + \
        str(config.IMAGE_HEIGHT) + "\n"
    auditString = auditString + "MAX_TEXT_LENGTH: " + \
        str(config.MAX_TEXT_LENGTH) + "\n"
    auditString = auditString + "RESIZE_IMAGE: " + \
        str(config.RESIZE_IMAGE) + "\n"
    auditString = auditString + "CONVERT_IMAGE_TO_MONOCHROME: " + \
        str(config.CONVERT_IMAGE_TO_MONOCHROME) + "\n"
    auditString = auditString + "MONOCHROME_BINARY_THRESHOLD: " + \
        str(config.MONOCHROME_BINARY_THRESHOLD) + "\n"
    auditString = auditString + "AUGMENT_IMAGE: " + \
        str(config.AUGMENT_IMAGE) + "\n\n"

    return auditString


def get_execution_log(paraCharSuccessRate, paraWordsSuccessRate):
    auditString = "Start Execution Time : " + \
        startTime.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
    auditString = auditString + "End Execution Time  :" + \
        datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + "\n"
    auditString = auditString + "Accumulated Processing Time : " + \
        str(config.ACCUMULATED_PROCESSING_TIME / 60) + " minutes" + "\n"
    auditString = auditString + "Characters Success Rate: " + \
        str(paraCharSuccessRate * 100.0) + "%\n"
    auditString = auditString + "Words Success Rate: " + \
        str(paraWordsSuccessRate * 100.0) + "%\n\n"

    return auditString


def accumulateProcessingTime(paraTimeSnapshot):
    config.ACCUMULATED_PROCESSING_TIME = config.ACCUMULATED_PROCESSING_TIME + \
        (time.time() - paraTimeSnapshot)


def main():

    if config.OPERATION_TYPE != config.OperationType.Infer:
        dataGenerator.LoadData(config.OPERATION_TYPE)

    if config.OPERATION_TYPE == config.OperationType.Training:
        auditString = "EXPERIMENT_NAME: " + config.EXPERIMENT_NAME + "\n"
        auditString = auditString + "Training Using Dataset: " + \
            str(config.OPERATION_TYPE) + "\n"
        print(auditString)
        config.auditLog(auditString)

        model = Model(config.DECODER_TYPE, mustRestore=False, dump=False)
        train(model)
    elif config.OPERATION_TYPE == config.OperationType.Validation or config.OPERATION_TYPE == config.OperationType.Testing:
        auditString = "EXPERIMENT_NAME: " + config.EXPERIMENT_NAME + "\n"
        auditString = auditString + "Validation/Tesing Using Dataset: " + \
            str(config.OPERATION_TYPE) + "\n"
        print(auditString)

        model = Model(config.DECODER_TYPE, mustRestore=True, dump=False)
        charErrorRate, charSuccessRate, wordsSuccessRate = validate(
            model, config.OPERATION_TYPE)

        auditString = auditString + \
            get_execution_log(charSuccessRate, wordsSuccessRate) + "\n"
        print(auditString)

        config.auditLog(auditString)

    elif config.OPERATION_TYPE == config.OperationType.Infer:  # infer text on test image
        print(open(config.fnResult).read())
        #model = Model(open(config.fnCharList, encoding="utf-8").read(), decoderType, mustRestore=True, dump=args.dump)
        model = Model(config.DECODER_TYPE, mustRestore=True, dump=False)
        inferSingleImage(model, config.fnInfer)


if __name__ == '__main__':
    main()
