import math
import numpy as np
import torch
import evaluation
import torch.nn as nn
import torch.optim as optim
import shutil
import os
import fileio

def train_model(X_train, Y_train, X_test, Y_test, NetClass, batch_size=64, num_epochs=100,
                lr=0.0001, weight_decay=0,
                device=torch.device("cpu"), seed=None,
                print_every=1, save_every=10, resume=None):
    
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    net = NetClass()
    net.to(device)
    
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    
    num_batches = math.ceil(X_train.size()[0] / batch_size)

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
        
        shuffle = np.arange(X_train.shape[0])
        np.random.shuffle(shuffle)
        X_train = X_train[shuffle]
        Y_train = Y_train[shuffle]

        losses = []
        scores = []

        for batch_num in range(num_batches):
            bottom = batch_size * batch_num
            top = min(batch_size * (batch_num + 1), X_train.size()[0])
            
            X_batch = X_train[bottom : top]
            X_batch = X_batch.to(device)
            Y_batch = Y_train[bottom : top]
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()

            Y_hat = net(X_batch)
            loss = criterion(Y_hat, Y_batch)
            loss.backward()
            optimizer.step()

            losses.append(loss.item() * X_batch.size()[0])
            scores.extend(evaluation.get_scores(Y_batch, np.argmax(Y_hat.data, axis=1)))

            
        test_score = np.mean(evaluation.get_scores(Y_test, np.argmax(net(X_test).data, axis=1)))
        is_best = False
        if test_score > best_score:
            best_score = test_score
            is_best = True
            
        this_loss = np.sum(losses) / X_train.size()[0]
        global_losses.append(this_loss)
            
        if epoch % print_every == 0:
            print('epoch ' + str(epoch) + ' loss: ' + str(this_loss))
            print('Train accuracy = ' + str(np.mean(scores)))
            print('Test accuracy = ' + str(test_score))
            
        if epoch % save_every == 0 or is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': net.state_dict(),
                'best_score': best_score,
                'global_losses': global_losses,
                'optimizer' : optimizer.state_dict(),
            }, is_best, 'checkpoint.pth.tar'.format(net.__class__.__name__))
            if is_best:
                filename = 'model_best.pkl'
                torch.save(mdl, '{}/{}'.format(fileio.OUTPUT_PREFIX, filename))
    print('Done')
    
    return global_losses



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.exists(fileio.OUTPUT_PREFIX):
        os.makedirs(fileio.OUTPUT_PREFIX)
        
    torch.save(state, '{}/{}'.format(fileio.OUTPUT_PREFIX, filename))
    
    if is_best:
        shutil.copyfile('{}/{}'.format(fileio.OUTPUT_PREFIX, filename), '{}/model_best.pth.tar'.format(fileio.OUTPUT_PREFIX))