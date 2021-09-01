from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import tensorflow as tf
import Config as config


class Model:
    "minimalistic TF model for HTR"

    def __init__(self, decoderType = config.DecoderType.BestPath, mustRestore=False, dump=False):
        "init model: add CNN, RNN and CTC and initialize TF"
        self.dump = dump
        self.charList = open(config.fnCharList, encoding="utf-8").read()
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID = 0

        # Whether to use normalization over a batch or a population
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        # input image batch
        self.inputImgs = tf.placeholder(tf.float32, shape=(
            None, config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

        # setup CNN, RNN and CTC
        self.setup5LayersCNN()
        self.setupRNN()
        self.setupCTC()

        # setup optimizer to train NN
        self.batchesTrained = 0
        self.learningRate = tf.placeholder(tf.float32, shape=[])
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.RMSPropOptimizer(
                self.learningRate).minimize(self.loss)

        self.auditModelDetails()
        # initialize TF
        (self.sess, self.saver) = self.setupTF()

    def auditModelDetails(self):
        total_parameters = 0
        saveString = "Model Details" + "\n"
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            saveString = saveString + "Shape:" + \
                str(shape) + " ,shape length:" + str(len(shape))
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            saveString = saveString + " , parameters: " + \
                str(variable_parameters) + "\n"
            total_parameters += variable_parameters

        saveString = saveString + "Total Parameters: " + \
            str(total_parameters) + "\n\n"

        print(saveString)
        config.auditLog(saveString)

    def setup5LayersCNN(self):
        "create CNN layers and return output of these layers"
        cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)
        pool = cnnIn4d  # input to first CNN layer

        # # list of parameters for the layers
        # kernelVals = [5, 5, 3, 3, 3]
        # featureVals = [1, 32, 64, 128, 128, 256]
        # strideVals = poolVals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        # numLayers = len(strideVals)

        # create layers
        # for i in range(numLayers):
        #     kernel = tf.Variable(
        #         tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
        #     conv = tf.nn.conv2d(pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
        #     conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
        #     relu = tf.nn.relu(conv_norm)
        #     pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1),
        #                           (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')

        self.kernel1 = tf.Variable(
            tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
        self.conv1 = tf.nn.conv2d(
            pool, self.kernel1, padding='SAME', strides=(1, 1, 1, 1))
        conv_norm = tf.layers.batch_normalization(
            self.conv1, training=self.is_train)
        self.relu1 = tf.nn.relu(conv_norm)
        self.pool1 = tf.nn.max_pool(
            self.relu1, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

        kernel = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
        self.conv2 = tf.nn.conv2d(
            self.pool1, kernel, padding='SAME', strides=(1, 1, 1, 1))
        conv_norm = tf.layers.batch_normalization(
            self.conv2, training=self.is_train)
        relu = tf.nn.relu(conv_norm)
        pool = tf.nn.max_pool(relu, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
        self.conv3 = tf.nn.conv2d(
            pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
        conv_norm = tf.layers.batch_normalization(
            self.conv3, training=self.is_train)
        relu = tf.nn.relu(conv_norm)
        pool = tf.nn.max_pool(relu, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))
        self.conv4 = tf.nn.conv2d(
            pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
        conv_norm = tf.layers.batch_normalization(
            self.conv4, training=self.is_train)
        relu = tf.nn.relu(conv_norm)
        pool = tf.nn.max_pool(relu, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1))
        self.conv5 = tf.nn.conv2d(
            pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
        conv_norm = tf.layers.batch_normalization(
            self.conv5, training=self.is_train)
        relu = tf.nn.relu(conv_norm)
        pool = tf.nn.max_pool(relu, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

        self.cnnOut4d = pool

    def setupCNN7Layers(self):
        "create CNN layers and return output of these layers"
        cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)

        kernel1 = tf.Variable(tf.truncated_normal([3, 3, 1, 64], stddev=0.1))
        conv1 = tf.nn.conv2d(
            cnnIn4d, kernel1, padding='SAME', strides=(1, 1, 1, 1))
        pool1 = tf.nn.max_pool(conv1, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

        kernel2 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
        conv2 = tf.nn.conv2d(
            pool1, kernel2, padding='SAME', strides=(1, 1, 1, 1))
        pool2 = tf.nn.max_pool(conv2, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

        kernel3 = tf.Variable(tf.truncated_normal(
            [3, 3, 128, 256], stddev=0.1))
        conv3 = tf.nn.conv2d(
            pool2, kernel3, padding='SAME', strides=(1, 1, 1, 1))

        kernel4 = tf.Variable(tf.truncated_normal(
            [3, 3, 256, 256], stddev=0.1))
        conv4 = tf.nn.conv2d(
            conv3, kernel4, padding='SAME', strides=(1, 1, 1, 1))
        pool3 = tf.nn.max_pool(conv4, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

        kernel5 = tf.Variable(tf.truncated_normal(
            [3, 3, 256, 512], stddev=0.1))
        conv5 = tf.nn.conv2d(
            pool3, kernel5, padding='SAME', strides=(1, 1, 1, 1))
        batch_norm1 = tf.layers.batch_normalization(
            conv4, training=self.is_train)

        kernel6 = tf.Variable(tf.truncated_normal(
            [3, 3, 512, 512], stddev=0.1))
        conv6 = tf.nn.conv2d(batch_norm1, kernel6,
                             padding='SAME', strides=(1, 1, 1, 1))
        batch_norm2 = tf.layers.batch_normalization(
            conv6, training=self.is_train)
        pool4 = tf.nn.max_pool(batch_norm2, (1, 1, 2, 1),
                               (1, 1, 2, 1), 'VALID')

        kernel7 = tf.Variable(tf.truncated_normal(
            [2, 2, 512, 512], stddev=0.1))
        conv7 = tf.nn.conv2d(batch_norm1, kernel7,
                             padding='SAME', strides=(1, 1, 1, 1))

        self.cnnOut4d = conv7

    def setupRNN(self):
        "create RNN layers and return output of these layers"
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

        # basic cells which is used to build RNN
        numHidden = 256
        cells = [tf.contrib.rnn.LSTMCell(
            num_units=numHidden, state_is_tuple=True) for _ in range(2)]  # 2 layers

        # stack basic cells
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d,
                                                        dtype=rnnIn3d.dtype)

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.truncated_normal(
            [1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
        self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(
            value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])

    def setupCTC(self):
        "create CTC loss and decoder and return them"
        # BxTxC -> TxBxC
        self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])

        # ground truth text as sparse tensor
        self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]), tf.placeholder(tf.int32, [None]),
                                       tf.placeholder(tf.int64, [2]))

        # calc loss for batch
        self.seqLen = tf.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(
            tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen,
                           ctc_merge_repeated=True))

        # calc loss for each element to compute label probability
        self.savedCtcInput = tf.placeholder(
            tf.float32, shape=[config.MAX_TEXT_LENGTH, None, len(self.charList) + 1])
        self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput,
                                             sequence_length=self.seqLen, ctc_merge_repeated=True)

        # decoder: either best path decoding or beam search decoding
        if self.decoderType == config.DecoderType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(
                inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
        elif self.decoderType == config.DecoderType.BeamSearch:
            self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen,
                                                         beam_width=50, merge_repeated=False)
        elif self.decoderType == config.DecoderType.WordBeamSearch:
            # import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
            word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')

            # prepare information about language (dictionary, characters in dataset, characters forming words)
            chars = str().join(self.charList)
            wordChars = open(config.fnwordCharList).read().splitlines()[0]
            corpus = open(config.fnCorpus).read()

            # decode using the "Words" mode of word beam search
            self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(self.ctcIn3dTBC, dim=2), 50, 'Words',
                                                                    0.0, corpus.encode(
                                                                        'utf8'), chars.encode('utf8'),
                                                                    wordChars.encode('utf8'))

    def setupTF(self):
        "initialize TF"
        print('Python: ' + sys.version)
        print('Tensorflow: ' + tf.__version__)

        sess = tf.Session()  # TF session

        # saver saves model to file
        saver = tf.train.Saver(max_to_keep=config.MAXIMUM_MODELS_TO_KEEP)
        modelDir = config.MODEL_PATH
        latestSnapshot = tf.train.latest_checkpoint(
            modelDir)  # is there a saved model?

        # if model must be restored (for inference), there must be a snapshot
        if self.mustRestore and not latestSnapshot:
            raise Exception('No saved model found in: ' + modelDir)

        # load saved model if available
        if latestSnapshot:
            print('Init with stored values from ' + latestSnapshot)
            saver.restore(sess, latestSnapshot)
        else:
            print('Init with new values')
            sess.run(tf.global_variables_initializer())

        return (sess, saver)

    def toSparse(self, texts):
        "put ground truth texts into sparse tensor for ctc_loss"
        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

        # go over all texts
        for (batchElement, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            CharactersIndexesOflabels = [self.charList.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(CharactersIndexesOflabels) > shape[1]:
                shape[1] = len(CharactersIndexesOflabels)
            # put each label into sparse tensor
            for (i, label) in enumerate(CharactersIndexesOflabels):
                indices.append([batchElement, i])
                values.append(label)

        return (indices, values, shape)

    def decoderOutputToText(self, ctcOutput, batchSize):
        "extract texts from output of CTC decoder"

        # contains string of labels for each batch element
        encodedLabelStrs = [[] for i in range(batchSize)]

        # word beam search: label strings terminated by blank
        if self.decoderType == config.DecoderType.WordBeamSearch:
            blank = len(self.charList)
            for b in range(batchSize):
                for label in ctcOutput[b]:
                    if label == blank:
                        break
                    encodedLabelStrs[b].append(label)

        # TF decoders: label strings are contained in sparse tensor
        else:
            # ctc returns tuple, first element is SparseTensor
            decoded = ctcOutput[0][0]

            # go over all indices and save mapping: batch -> values
            idxDict = {b: [] for b in range(batchSize)}
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batchElement = idx2d[0]  # index according to [b,t]
                encodedLabelStrs[batchElement].append(label)

        # map labels to chars for all batch elements
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]

    def trainBatch(self, batch):
        "feed a batch into the NN to train it"
        numBatchElements = len(batch.imgs)
        sparse = self.toSparse(batch.gtTexts)
        rate = 0.01 if self.batchesTrained < 10 else (
            0.001 if self.batchesTrained < 10000 else 0.0001)  # decay learning rate
        evalList = [self.optimizer, self.loss]
        feedDict = {self.inputImgs: batch.imgs, self.gtTexts: sparse,
                    self.seqLen: [config.MAX_TEXT_LENGTH] * numBatchElements, self.learningRate: rate,
                    self.is_train: True}

        (_, lossVal) = self.sess.run(evalList, feedDict)

        self.batchesTrained += 1
        return lossVal

    def dumpNNOutput(self, rnnOutput):
        "dump the output of the NN to CSV file(s)"
        dumpDir = '../dump/'
        if not os.path.isdir(dumpDir):
            os.mkdir(dumpDir)

        # iterate over all batch elements and create a CSV file for each one
        maxT, maxB, maxC = rnnOutput.shape
        for b in range(maxB):
            csv = ''
            for t in range(maxT):
                for c in range(maxC):
                    csv += str(rnnOutput[t, b, c]) + ';'
                csv += '\n'
            fn = dumpDir + 'rnnOutput_' + str(b) + '.csv'
            print('Write dump of NN to file: ' + fn)
            with open(fn, 'w') as f:
                f.write(csv)

    def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
        "feed a batch into the NN to recognize the texts"

        # decode, optionally save RNN output
        numBatchElements = len(batch.imgs)
        evalRnnOutput = self.dump or calcProbability
        evalList = [self.decoder] + \
            ([self.ctcIn3dTBC] if evalRnnOutput else [])
        feedDict = {self.inputImgs: batch.imgs, self.seqLen: [config.MAX_TEXT_LENGTH] * numBatchElements,
                    self.is_train: False}

        evalRes = self.sess.run(evalList, feedDict)

        #####################################################
#         result = self.sess.run(self.cnnOut4d, feedDict)
#         result = np.squeeze(result)
#
        # result[result<0]=0
        # savetxt('result.csv', result, delimiter=',')
        ###writer = tf.summary.FileWriter('d://tensorflow_example',self.sess.graph)

#        filters = result.shape[2]
#         plt.figure(1, figsize=(40, 40))
#         n_columns = 1
#         n_rows = 1 # math.ceil(filters / n_columns) + 1
#
#         plt.ion()
#         #for i in range(filters):
#         #    plt.subplot(n_rows, n_columns, i + 1)
#         #    plt.title('Filter ' + str(i))
#         plt.imshow(result[:], interpolation="nearest", cmap="gray")
#         plt.show()
#         return
#         ######################################################

        decoded = evalRes[0]
        texts = self.decoderOutputToText(decoded, numBatchElements)

        # feed RNN output and recognized text into CTC loss to compute labeling probability
        probs = None
        if calcProbability:
            sparse = self.toSparse(
                batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
            ctcInput = evalRes[1]
            evalList = self.lossPerElement
            feedDict = {self.savedCtcInput: ctcInput, self.gtTexts: sparse,
                        self.seqLen: [config.MAX_TEXT_LENGTH] * numBatchElements, self.is_train: False}

            lossVals = self.sess.run(evalList, feedDict)

            probs = np.exp(-lossVals)

        # dump the output of the NN to CSV file(s)
        if self.dump:
            self.dumpNNOutput(evalRes[1])

        return (texts, probs)

    def save(self):
        "save model to file"
        self.snapID += 1
        self.saver.save(self.sess, config.MODEL_PATH +
                        config.EXPERIMENT_NAME, global_step=self.snapID)
