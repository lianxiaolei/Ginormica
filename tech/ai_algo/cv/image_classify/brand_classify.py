# -*- coding: utf-8 -*-

import cv2
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
import sys


class Evaluator(Callback):
    def __init__(self, model):
        self.accs = []
        self.model = model

    def on_epoch_end(self, epoch, logs={}):
        print('model--------------------------', self.model.evaluate(steps=10))
        acc = self.model.evaluate(steps=1) * 100
        self.accs.append(acc)
        print('acc: %f%%' % acc)


class BrandClassify(object):

    def __init__(self, batch_size=64, gene=1, width=600, height=300, num_class=100):
        self.datagen = image.ImageDataGenerator(featurewise_center=False,
                                                samplewise_center=False,
                                                featurewise_std_normalization=False,
                                                samplewise_std_normalization=False,
                                                zca_whitening=False,
                                                rotation_range=0.,
                                                width_shift_range=0.,
                                                height_shift_range=0.,
                                                shear_range=0.,
                                                zoom_range=0.,
                                                channel_shift_range=0.,
                                                fill_mode='nearest',
                                                cval=0.0,
                                                horizontal_flip=False,
                                                vertical_flip=False,
                                                rescale=None,
                                                preprocessing_function=None,
                                                data_format=K.image_data_format(),
                                                )
        self.labels = None
        self.files = None
        self.batch_size = batch_size
        print('self batch size', self.batch_size)
        self.gene = gene
        self.width = width
        self.height = height
        self.num_class = num_class
        self.base_model = None
        self.model = None

    def read_data(self, flag, path='../../../../assets/brand_images'):
        """
        read the dataset，dir format must be: ./train/, ./train.txt, ./test/, ./test.txt
        :param flag: 'train' or 'test'
        :param path
        """
        content = open(os.path.join(path, '%s.txt' % flag))

        imgs = []
        labels = []

        lines = content.readlines()
        # tqdm is the progress bar, please install it with "pip install tqdm".
        # if u wan't, you can replace tqdm(lines) with lines.
        for i in lines:
            fname, y = i.replace('\n', '').split(' ')
            y = int(y)

            x = os.path.join(path, flag, fname)

            imgs.append(x)
            labels.append(y)
        return np.array(imgs), np.array(labels) - 1

    def data_gen(self, batch_size=128, gene=4):
        """
        generate the data with the size batch_size
        :param batch_size:
        :param gene:
        :return:
        """
        X_ = np.zeros([batch_size, self.height, self.width, 3], np.uint8)
        y_ = np.zeros([batch_size, 1], np.uint8)


        while True:

            index = np.random.choice(len(self.labels), batch_size, replace=False)
            label_list = self.labels[index]
            file_list = self.files[index]

            for i in range(len(file_list)):
                fname = file_list[i]
                img = cv2.imread(fname)
                img = cv2.resize(img, (self.width, self.height))
                img = img / 255.0
                X_[i] = img
                y_[i] = np.array(label_list[i])

            # for fname in file_list:
            #     img = cv2.imread(fname)
            #     img = cv2.resize(img, (self.width, self.height))
            #     X_.append(img.tolist())
            #
            # X_ = np.array(X_)
            # y_ = np.array(label_list)

            i = 0
            X = np.zeros([batch_size * gene, self.height, self.width, 3], np.uint8)
            y = np.zeros([batch_size * gene, 1], np.uint8)
            for batch in self.datagen.flow(X_, y_, batch_size=batch_size):
                X[i * batch_size: (i + 1) * batch_size] = batch[0]
                y[i * batch_size: (i + 1) * batch_size] = batch[1]

                # if not type(X) == np.ndarray:
                #     X = batch[0]
                #     y = batch[1]
                # else:
                #     X = np.concatenate([X, batch[0]], axis=0)
                #     y = np.concatenate([y, batch[1]], axis=0)

                i += 1
                if i >= gene:
                    break
            #
            # # X = np.array(X)
            # y_onehot = to_categorical(np.array(y), num_classes=100)
            # print('input :', X.dtype, X.shape, sys.getsizeof(X),
            #       y_onehot.dtype, y_onehot.shape, sys.getsizeof(y_onehot))
            #
            # yield X, y_onehot

            yield X, to_categorical(np.array(y), num_classes=100)
            # return

    def model_struct(self):
        input_tensor = Input((self.height, self.width, 3))
        # vgg = VGG16(weights='../../../../models/VGG16_WEIGHTS.h5', include_top=False, input_tensor=input_tensor)
        # print('the last vgg layer is ', vgg.layers[-1])
        # print('the last vgg layer output is ', vgg.output)

        # tensor_shape = vgg.output.shape
        # print(tensor_shape)

        # rnn_length = tensor_shape[1].value
        # rnn_dimen = tensor_shape[2].value * tensor_shape[3].value
        # units = tensor_shape[3].value
        # print(rnn_length, rnn_dimen, units)
        x = input_tensor
        for i in range(3):
            # x = Conv2D(100, (3, 3), kernel_initializer='he_normal')(x)
            # # x = BatchNormalization()(x)
            # x = Activation('relu')(x)
            x = Conv2D(100, (3, 3), kernel_initializer='he_normal')(x)
            # x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPool2D(pool_size=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(400, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(100, kernel_initializer='he_normal', activation='softmax')(x)

        self.base_model = Model(input=input_tensor, output=x)
        # labels = Input(name='labels', shape=[self.num_class], dtype='float32')
        self.model = Model(inputs=input_tensor, outputs=x)

    def train(self):
        self.files, self.labels = self.read_data('train')
        # print('读取数据', len(self.files), len(self.labels))

        self.model_struct()
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        # print('model:', self.model)
        self.model.fit_generator(
            self.data_gen(self.batch_size, self.gene),
            steps_per_epoch=200,
            epochs=20,
            max_q_size=1,
            # callbacks=[Evaluator(self.model)],
            validation_data=self.data_gen(20, 4),
            validation_steps=10
        )
        print(self.model.output)
        self.base_model.save('brand_classify_vgg16.h5')
        self.base_model.save_weights('brand_classify_vgg16_weights.h5')

    def evaluate(self, steps=10):
        print('------------------steps-------------------', steps)
        batch_acc = 0
        generator = self.data_gen(self.batch_size, self.gene)
        for i in range(steps):
            X_test, y_test = next(generator)
            y_pred = self.base_model.predict(X_test)

            acc = np.mean(np.argmax(y_pred) == np.argmax(y_test))
            batch_acc += acc

        return batch_acc / steps


if __name__ == '__main__':
    bc = BrandClassify(batch_size=100, gene=4, width=150, height=80)
    bc.train()
    print('done')
