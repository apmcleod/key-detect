#!/usr/bin/env python

import argparse
import math
import numpy as np
import torch
import evaluation
import torch.nn as nn
import torch.optim as optim
import shutil
import os
import sys
import fileio
import h5py


def train_model(h5_file, net, model_name, batch_size=64, num_epochs=100,
                optimizer=None, scheduler=None, log_path=None, device=torch.device("cpu"),
                seed=None, print_every=1, save_every=10, resume=None,
                small_ram=False):

    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    net.to(device)

    X_test_full = None
    Y_test_full = None
    X_test_flat = None
    X_test_full = None
    
    data_file = h5py.File(h5_file, 'r')
    train_size = data_file['X_train'].shape[0]
    test_size = data_file['X_test'].shape[0]
    if not small_ram:
        X_train_full = data_file['X_train'].value
        Y_train_full = data_file['Y_train'].value
        X_test_full = data_file['X_test'].value
        Y_test_full = data_file['Y_test'].value
        # Make a test set with an even number of each key
        unique, counts = np.unique(Y_test_full, return_counts=True)
        nr_samples = np.min(counts)
        selected = {}
        for ii in range(24):
            these_idx = np.where(Y_test_full==ii)[0]
            selected[ii] = np.random.choice(these_idx, nr_samples, replace=False)
        idx_list = np.hstack(selected.values())
        X_test_flat = X_test_full[idx_list]
        Y_test_flat = Y_test_full[idx_list]
    data_file.close()

    num_batches = math.ceil(train_size / batch_size)

    criterion = nn.CrossEntropyLoss()
    if not optimizer:
        optimizer = optim.Adam(net.parameters())

    epoch_start = 0
    best_score = 0
    global_train_losses = []
    global_test_losses = []

    if resume:
        print("loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        epoch_start = checkpoint['epoch']
        net.load_state_dict(checkpoint['model'])
        net.to(device)
        best_score = checkpoint['best_score']
        global_test_losses = checkpoint['global_test_losses']
        global_train_losses = checkpoint['global_train_losses']
        optimizer.load_state_dict(checkpoint['optimizer'])

    print(net)

    for epoch in range(epoch_start, num_epochs):
        if log_path is None:
            loss_log_file = sys.stderr
            acc_log_file = sys.stderr
        elif epoch==0:
            loss_log_file = open('{}_loss.log'.format(log_path), "w")
            acc_log_file = open('{}_acc.log'.format(log_path), "w")
            print('Train,Test,Test (flat)', file=loss_log_file)
            print('Train,Test,Test (flat)', file=acc_log_file) 
        else:
            loss_log_file = open('{}_loss.log'.format(log_path), "a")
            acc_log_file = open('{}_acc.log'.format(log_path), "a")
        
        
        shuffle = np.arange(train_size)
        np.random.shuffle(shuffle)

        avg_loss = 0
        avg_score = 0

        for batch_num in range(num_batches):
            if (batch_num + 1) % 50 == 0:
                print('Batch {}/{}'.format(batch_num + 1, num_batches))

            bottom = batch_size * batch_num
            top = min(batch_size * (batch_num + 1), train_size)

            if small_ram:
                indeces = sorted(shuffle[bottom : top])
                data_file = h5py.File(h5_file, 'r')
                X_batch = torch.from_numpy(data_file['X_train'][indeces]).float().to(device)
                Y_batch = torch.from_numpy(data_file['Y_train'][indeces]).to(device)
                data_file.close()
            else:
                indeces = shuffle[bottom : top]
                X_batch = torch.from_numpy(X_train_full[indeces]).float().to(device)
                Y_batch = torch.from_numpy(Y_train_full[indeces]).to(device)

            optimizer.zero_grad()

            Y_hat = net(X_batch)
            loss = criterion(Y_hat, Y_batch)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() * X_batch.size()[0]
            avg_score += np.sum(evaluation.get_scores(Y_batch, np.argmax(Y_hat.data, axis=1)))

        avg_loss = avg_loss / train_size
        avg_score = avg_score / train_size
        test_score, test_loss = get_score_batched(net, h5_file, X_test_full, Y_test_full, 
                                                  criterion, small_ram=small_ram, device=device)
        flat_score, flat_loss = get_score_batched(net, h5_file, X_test_flat, Y_test_flat, 
                                                  criterion, small_ram=small_ram, device=device)

        if scheduler:
            scheduler.step(test_score)

        print("{},{},{}".format(avg_loss, test_loss, flat_loss), file=loss_log_file)
        print("{},{},{}".format(avg_score, test_score, flat_score), file=acc_log_file)
        if log_path is not None:
            loss_log_file.close()
            acc_log_file.close()

        is_best = False
        if test_score > best_score:
            best_score = test_score
            is_best = True

        global_train_losses.append(avg_loss)
        global_test_losses.append(test_loss)

        if epoch % print_every == 0:
            print('epoch ' + str(epoch) + ' loss: ' + str(avg_loss))
            print('Train accuracy = ' + str(avg_score))
            print('Test accuracy = ' + str(test_score))

        if epoch % save_every == 0 or is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': net.state_dict(),
                'best_score': best_score,
                'global_test_losses': global_test_losses,
                'global_train_losses': global_train_losses,
                'optimizer' : optimizer.state_dict(),
            }, is_best, model_name, 'checkpoint.pth.tar')
            if is_best:
                filename = '{}_model_best.pkl'.format(model_name)
                torch.save(net, '{}/{}'.format(fileio.OUTPUT_PREFIX, filename))
    print('Done')

    return global_train_losses, global_test_losses


def get_score_batched(net, h5_file, X_full, Y_full, criterion, small_ram=False,
                      device=torch.device("cpu"), batch_size=256, X='X_test', Y='Y_test'):

    if small_ram:
        data_file = h5py.File(h5_file, 'r')
        X_size = data_file[X].shape[0]
        data_file.close()
    else:
        X_size = X_full.shape[0]

    num_batches = math.ceil(X_size / batch_size)
    avg_score = 0
    avg_loss = 0

    for batch_num in range(num_batches):
        bottom = batch_size * batch_num
        top = min(batch_size * (batch_num + 1), X_size)

        if small_ram:
            data_file = h5py.File(h5_file, 'r')
            X_batch = torch.from_numpy(data_file[X][bottom : top]).float().to(device)
            Y_batch = torch.from_numpy(data_file[Y][bottom : top]).to(device)
            data_file.close()
        else:
            X_batch = torch.from_numpy(X_full[bottom : top]).float().to(device)
            Y_batch = torch.from_numpy(Y_full[bottom : top]).to(device)

        Y_hat = net(X_batch)
        loss = criterion(Y_hat, Y_batch)
        loss.backward()

        avg_loss += loss.item() * X_batch.size()[0]

        score = np.sum(evaluation.get_scores(Y_batch, np.argmax(Y_hat.data, axis=1)))
        avg_score += score

    return avg_score / X_size, avg_loss / X_size


def save_checkpoint(state, is_best, model_name, filename='checkpoint.pth.tar'):
    if not os.path.exists(fileio.OUTPUT_PREFIX):
        os.makedirs(fileio.OUTPUT_PREFIX)

    torch.save(state, '{}/{}_{}'.format(fileio.OUTPUT_PREFIX, model_name, filename))

    if is_best:
        shutil.copyfile('{}/{}_{}'.format(fileio.OUTPUT_PREFIX, model_name, filename),
                        '{}/{}_model_best.pth.tar'.format(fileio.OUTPUT_PREFIX, model_name))



if __name__ == '__main__':
    import full_model

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", "--model", help="Which model to use for prediction",
                        type=int, choices=range(1, 3), required=True)
    parser.add_argument("-s", "--seed", help="A seed to use for modelling",
                        type=int, default=None)
    parser.add_argument("-c", "--cuda", help="Cuda device number to use",
                        type=str, default="0")
    parser.add_argument("-n", "--num-epochs", help="Number of epochs to train",
                        type=int, default=100)
    args = parser.parse_args()

    model = args.model
    if model == 1:
        model_name = 'MO1'
    elif model == 2:
        model_name = 'MO2'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed = args.seed
    if seed:
        torch.manual_seed(seed)

    h5_file = 'data/working/data.h5'

    if model_name == 'MO2':
        net = full_model.ReproductionNet()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,
                              weight_decay=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, threshold=0.0)
    else:
        model_name = 'MO1'
        net = full_model.ConvBiLstm()
        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, threshold=0.0)

    log_dir = 'data/output/logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = "{}/{}_trn_tst".format(log_dir, model_name)
    train_losses, test_losses = train_model(h5_file, net, model_name,
                                            num_epochs=args.num_epochs,
                                            optimizer=optimizer, scheduler=scheduler,
                                            device=device, seed=seed,
                                            log_path=log_path)



