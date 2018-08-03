import math
import numpy as np
import torch
import evaluation
import torch.nn as nn
import torch.optim as optim

def train_model(X_train, Y_train, X_test, Y_test, net, batch_size=64, num_epochs=1000,
                lr=0.0001, weight_decay=0,
                device=torch.device("cpu"), print_every=5):
    net.to(device)
    print(net)
    
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    
    num_batches = math.ceil(X_train.size()[0] / batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):

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

        if epoch % print_every == 0:
            print('epoch ' + str(epoch) + ' loss: ' + str(np.sum(losses) / X_train.size()[0]))
            print('Train accuracy = ' + str(np.mean(scores)))
            print('Test accuracy = ' + str(np.mean(evaluation.get_scores(Y_test, np.argmax(net(X_test).data, axis=1)))))

    print('Done')