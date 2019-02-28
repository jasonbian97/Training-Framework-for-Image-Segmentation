import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
# customized import
# from [filename] import [function name]
from .load import *
from .utils import split_train_val, batch
from .dice_loss import *
###########################################################################
#                               Train                                     #
###########################################################################

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.1,
              save_cp=True,
              gpu=True,
              img_scale=1.,
              dir_img,
              dir_mask,
              dir_checkpoint):
    # return only the file name without extension.
    ids = get_ids(dir_img)
    # generate tuples like (id,#)
    ids = split_ids(ids)
    # split into train and val w.r.t val_percent
    iddataset = split_train_val(ids, val_percent)
    # show configeration info
    print('''
        Starting training:
            Epochs: {}
            Batch size: {}
            Learning rate: {}
            Training size: {}
            Validation size: {}
            Checkpoints: {}
            CUDA: {}
        '''.format(epochs, batch_size, lr, len(iddataset['train']),
                   len(iddataset['val']), str(save_cp), str(gpu)))
    # use N_train to show where the training process is
    N_train = len(iddataset['train'])
    # set optimizer and criterion for loss
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)
    criterion = nn.BCELoss()
    # start epoch loop
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        # set net mode to "train"
        net.train()
        # preprocess image in your dataset/dir, including:
        # 1. resize and crop
        # 2. transform image from HWC to CHW
        # 3. normalize images e.g. img/255.
        # 4. convert GT image into binary
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)
        #
        epoch_loss = 0
        # start batch loop
        for i, b in enumerate(batch(train, batch_size)):
            # generate tensor batch
            imgs_list=[]
            mask_list=[]
            for k in b:
                imgs_list.append(k[0])
                mask_list.append(k[1])
            imgs = np.array(imgs_list).astype(np.float32)
            true_masks = np.array(mask_list).astype(np.float32)
            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)
            # The shape here is (Batchsize,C,W,H)
            # Load to training data to GPU
            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()
            # inference once
            masks_pred = net(imgs)
            # flatten 4D matrix into 1D for computation simplicity
            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat = true_masks.view(-1)
            # calculate loss
            loss = criterion(masks_probs_flat, true_masks_flat)
            # accumulate loss of each batch to calculate epoch loss
            epoch_loss += loss.item()
            # show where the training process is
            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            # clear the place for new grad
            optimizer.zero_grad()
            # BP
            loss.backward()
            # update optimizer to possibly change learning rate
            optimizer.step()
        # show loss os this Epoch
        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        # test model using val set
        val_dice = eval_net(net, val, gpu)
        print('Validation Dice Coeff: {}'.format(val_dice))
        # save model in dir_checkpoint
        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))

###########################################################################
#                               Evaluate                                  #
###########################################################################

def eval_net(net, dataset, gpu=False):
    # set model to evaluation model
    net.eval()
    tot = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]
        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        # The shape here is (1,1,W,H)
        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)[0]
        mask_pred = (mask_pred > 0.5).float()
        # compute coefficient value
        tot += dice_coeff(mask_pred, true_mask).item()
    return tot / i

###########################################################################
#                               get_args                                  #
###########################################################################
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=20, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    # parser.add_option('-s', '--scale', dest='scale', type='float',
    #                   default=1., help='downscaling factor of the images')
    # Directory
    parser.add_option('-dir_img', '--scale', dest='dir_img', type='string',
                      default='/home/jasonbian/Desktop/WholeDataSet/DataSet1.0/cropped/')
    parser.add_option('-dir_mask', '--scale', dest='dir_mask', type='string',
                      default='/home/jasonbian/Desktop/WholeDataSet/DataSet1.0/BinaryMask3/')
    parser.add_option('-dir_checkpoint', '--scale', dest='dir_checkpoint', type='string',
                      default='./checkpoints/')

    (options, args) = parser.parse_args()
    return options

###########################################################################
#                               Main                                      #
###########################################################################
if __name__ == '__main__':
    # get global parameters: including: epochs,batchsize,lr,gpu
    args = get_args()
    # generate Model instance
    net = UNet(n_channels=3, n_classes=1)
    # read global paprameters and do some preparation
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory
    # read args and train Model
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    #  save interrupt model
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

