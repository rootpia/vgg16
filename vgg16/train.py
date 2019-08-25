#!/usr/bin/env python

from __future__ import print_function

import argparse
import numpy as np
from PIL import Image
import time
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


# Network definition
class VGG16Model(chainer.Chain):

    def __init__(self, out_size):
        super(VGG16Model, self).__init__(
            base = L.VGG16Layers(),
            fc = L.Linear(None, out_size)
        )

    def __call__(self, x):
        h = self.base(x, layers=['fc7'])
        y = self.fc(h['fc7'])
        return y

def main():
    parser = argparse.ArgumentParser(description='vgg16')
    parser.add_argument('--input', '-i', type=str, default='./images/cat.jpg',
                        help='predict imagefile')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--classes', '-c', type=int, default=5,
                        help='Number of classes')
    args = parser.parse_args()

    print('# GPU: {}'.format(args.gpu))
    print('# classes: {}'.format(args.classes))
    print('# batch: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # prepare network
    model = L.Classifier(VGG16Model(out_size=args.classes))
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    optimizer = chainer.optimizers.Adam(alpha=0.0001)
    optimizer.setup(model)
    # training rate
    for func_name in model.predictor.base._children:
        for param in model.predictor.base[func_name].params():
            param.update_rule.hyperparam.alpha *= 0.1

    # Transfer learning
    img = Image.open(args.input)
    x = L.model.vision.vgg.prepare(img)
    x = x[np.newaxis] # batch size
    print('predict')
    starttime = time.time()
    result = model.predictor(x)
    print(result)
    predict = F.argmax(F.softmax(result, axis=1), axis=1)
    endtime = time.time()
    print(predict, (endtime - starttime), 'sec')  # variable([0]) 45.696336031 sec
    exit()


    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
#    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    # save model
    chainer.serializers.save_npz(args.out + '/pretrained_model', model_mlp)


if __name__ == '__main__':
    main()
