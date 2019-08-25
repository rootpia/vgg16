#!/usr/bin/env python

import argparse

import chainer
import numpy as np

from train_mnist import MLP

def print_predict(model):
    # Load the MNIST dataset
    _, test = chainer.datasets.get_mnist()
    test, label = chainer.dataset.concat_examples(test)

    pred = model(test)
    pred = chainer.functions.softmax(pred).data
    label_y = [np.argmax(pred[i]) for i in range(len(pred))]

    for ii in range(1):
        print('-------------------------------')
        print('gt   :{0}'.format(label[ii]))
        print('pred :{0}'.format(label_y[ii]))
        print('percentage:')
        for jj in range(10):
            print('[{0}]: {1:1.3f}'.format(jj, pred[ii][jj]))

def single_predictor(model, image):
    test = np.array(image).reshape(1, -1)
    pred = model(test)
    pred = chainer.functions.softmax(pred).data
    label_y = [np.argmax(pred[i]) for i in range(len(pred))]
    return (pred[0], label_y[0])

def seq_predictor(model):
    # Load the MNIST dataset
    _, test = chainer.datasets.get_mnist()
    test, label = chainer.dataset.concat_examples(test)

    for ii in range(2):
        pred = single_predictor(model, test[ii])

        print('-------------------------------')
        print('gt   :{0}'.format(label[ii]))
        print('pred :{0}'.format(pred[1]))
        print('percentage:')
        for jj in range(10):
            print('[{0}]: {1:1.3f}'.format(jj, pred[0][jj]))

def main():
    parser = argparse.ArgumentParser(description='regression of kWh')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of units')
    args = parser.parse_args()

    # load model
    model = MLP(args.unit, 10)
    chainer.serializers.load_npz(args.out + '/pretrained_model', model)

#    print_predict(model)
    seq_predictor(model)


if __name__ == '__main__':
    main()

