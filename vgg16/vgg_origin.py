#!/usr/bin/env python

from __future__ import print_function

import argparse
import numpy as np
import time
from PIL import Image
import chainer
import chainer.functions as F
import chainer.links as L


def main():
    parser = argparse.ArgumentParser(description='vgg16')
    parser.add_argument('--input', '-i', type=str, default='./images/cat.jpg',
                        help='predict imagefile')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('Input: {}'.format(args.input))
    print('')

    # import VGG model
    print('load network')
    vgg16 = L.VGG16Layers()
    print('load network, done.')

    # prediction test
    img = Image.open(args.input)
    x = L.model.vision.vgg.prepare(img)
    x = x[np.newaxis] # batch size
    print('predict')
    starttime = time.time()
    result = vgg16(x)
    predict = F.argmax(F.softmax(result['prob'], axis=1), axis=1)
    endtime = time.time()
    print(predict, (endtime - starttime), 'sec')  # variable([281]) 47.0666120052 sec


if __name__ == '__main__':
    main()
