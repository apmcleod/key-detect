import math
import numpy as np
import torch
import evaluation
import torch.nn as nn
import torch.optim as optim

def train_model(X_train, Y_train, X_test, Y_test, net, batch_size=64, num_epochs=1000, lr=0.0001):
    print(net)
    
    num_batches = math.ceil(X_train.size()[0] / batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(num_epochs):

        losses = []
        scores = []

        for batch_num in range(num_batches):
            bottom = batch_size * batch_num
            top = min(batch_size * (batch_num + 1), X_train.size()[0])
            X_batch = X_train[bottom : top]
            Y_batch = Y_train[bottom : top]

            optimizer.zero_grad()

            Y_hat = net(X_batch)
            loss = criterion(Y_hat, Y_batch)
            loss.backward()
            optimizer.step()

            losses.append(loss.item() * X_batch.size()[0])
            scores.extend(evaluation.get_scores(Y_batch.numpy(), np.argmax(Y_hat.data, axis=1).numpy()))

        if epoch % 5 == 0:
            print('epoch ' + str(epoch) + ' loss: ' + str(np.sum(losses) / X_train.size()[0]))
            print('Train accuracy = ' + str(np.mean(scores)))
            print('Test accuracy = ' + str(np.mean(evaluation.get_scores(Y_test.numpy(), np.argmax(net(X_test).data, axis=1).numpy()))))

    print('Done')
    
    
    
'''
Call like:

DATA_DIR = './data/working'

labels = pd.read_pickle("{}/labels.pkl".format(DATA_DIR))

with np.load("{}/splits.npz".format(DATA_DIR)) as splits:
    train_idx = splits['train_idx']
    test_idx = splits['test_idx']

X = np.load("{}/X_cqt.npz".format(DATA_DIR))['X']
X_train = X[train_idx, :]
X_test = X[test_idx, :]

Y = np.load("{}/Y.npz".format(DATA_DIR))['Y']
Y_train = Y[train_idx, :]
Y_test = Y[test_idx, :]

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()

Y_train = torch.from_numpy(np.argmax(Y_train, axis=1))
Y_test = torch.from_numpy(np.argmax(Y_test, axis=1))

model_tester.train_model(X_train, Y_train, X_test, Y_test, full_model.ConvLstm())
'''