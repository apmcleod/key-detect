import math
import numpy as np
import torch
import evaluation
import torch.nn as nn
import torch.optim as optim
import shutil
import os
import fileio
import h5py

def train_model(h5_file, net, model_name, batch_size=64, num_epochs=100,
                lr=0.0001, weight_decay=0,
                device=torch.device("cpu"), seed=None,
                print_every=1, save_every=10, resume=None):
    
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    net.to(device)
    
    data_file = h5py.File(h5_file, 'r')
    train_size = data_file['X_train'].shape[0]
    test_size = data_file['X_test'].shape[0]
    data_file.close()
    
    num_batches = math.ceil(train_size / batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    epoch_start = 0
    best_score = 0
    global_losses = []
    
    if resume:
        print("loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        epoch_start = checkpoint['epoch']
        net.load_state_dict(checkpoint['model'])
        net.to(device)
        best_score = checkpoint['best_score']
        global_losses = checkpoint['global_losses']
        optimizer.load_state_dict(checkpoint['optimizer'])
       
    print(net)

    for epoch in range(epoch_start, num_epochs):
        shuffle = np.arange(train_size)
        np.random.shuffle(shuffle)

        avg_loss = 0
        avg_score = 0

        for batch_num in range(num_batches):
            if batch_num % 20 == 0:
                print('Batch {}/{}'.format(batch_num, num_batches))
                
            bottom = batch_size * batch_num
            top = min(batch_size * (batch_num + 1), train_size)
            indeces = sorted(shuffle[bottom : top])
            
            data_file = h5py.File(h5_file, 'r')
            X_batch = torch.from_numpy(data_file['X_train'][indeces]).float()
            X_batch = X_batch.to(device)
            Y_batch = torch.from_numpy(data_file['Y_train'][indeces])
            Y_batch = Y_batch.to(device)
            data_file.close()

            optimizer.zero_grad()

            Y_hat = net(X_batch)
            loss = criterion(Y_hat, Y_batch)
            loss.backward()
            optimizer.step()

            score = np.sum(evaluation.get_scores(Y_batch, np.argmax(Y_hat.data, axis=1)))
            print('Batch {} loss = {}'.format(batch_num, loss.item()))
            print('Batch {} score = {}'.format(batch_num, score / X_batch.size()[0]))
            avg_loss += loss.item() * X_batch.size()[0]
            avg_score += score

        avg_loss = avg_loss / train_size
        avg_score = avg_score / train_size
        test_score = get_score_batched(net, h5_file, device=device)
        
        is_best = False
        if test_score > best_score:
            best_score = test_score
            is_best = True
            
        global_losses.append(avg_loss)
            
        if epoch % print_every == 0:
            print('epoch ' + str(epoch) + ' loss: ' + str(avg_loss))
            print('Train accuracy = ' + str(avg_score))
            print('Test accuracy = ' + str(test_score))
            
        if epoch % save_every == 0 or is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': net.state_dict(),
                'best_score': best_score,
                'global_losses': global_losses,
                'optimizer' : optimizer.state_dict(),
            }, is_best, model_name, 'checkpoint.pth.tar')
            if is_best:
                filename = '{}_model_best.pkl'.format(model_name)
                torch.save(net, '{}/{}'.format(fileio.OUTPUT_PREFIX, filename))
    print('Done')
    
    return global_losses


def get_score_batched(net, h5_file, device=torch.device("cpu"), batch_size=256, X='X_test', Y='Y_test'):
    data_file = h5py.File(h5_file, 'r')
    X_size = data_file[X].shape[0]
    data_file.close()
    
    num_batches = math.ceil(X_size / batch_size)
    avg_score = 0
    
    for batch_num in range(num_batches):
        bottom = batch_size * batch_num
        top = min(batch_size * (batch_num + 1), X_size)

        data_file = h5py.File(h5_file, 'r')
        X_batch = torch.from_numpy(data_file[X][bottom : top]).float()
        X_batch = X_batch.to(device)
        Y_batch = torch.from_numpy(data_file[Y][bottom : top])
        Y_batch = Y_batch.to(device)
        data_file.close()
        
        score = np.sum(evaluation.get_scores(Y_batch, np.argmax(net(X_batch).data, axis=1)))
        avg_score += score
        
        print('Batch {} score sum = {}'.format(batch_num, score))
        print('Running average = {}'.format(avg_score))
        
    print('final avg = {}'.format(avg_score / X_size))
    return avg_score / X_size


def save_checkpoint(state, is_best, model_name, filename='checkpoint.pth.tar'):
    if not os.path.exists(fileio.OUTPUT_PREFIX):
        os.makedirs(fileio.OUTPUT_PREFIX)
        
    torch.save(state, '{}/{}_{}'.format(fileio.OUTPUT_PREFIX, model_name, filename))
    
    if is_best:
        shutil.copyfile('{}/{}'.format(fileio.OUTPUT_PREFIX, filename), '{}/{}_model_best.pth.tar'.format(fileio.OUTPUT_PREFIX, model_name))