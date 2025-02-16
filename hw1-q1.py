#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):    
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """        
        y_hat = self.predict(x_i)
        # TODO nao atualiza!!!
        ## if y_hat!= 0:
        # #   print("self w",self.W, "y_i", y_i, "y_hat", y_hat)
        if y_hat != y_i:
            # self.W += y_i * x_i
            # multi class
            self.W[y_i, :] += x_i
            self.W[y_hat, :] -= x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b
        
        # Get probability scores according to the model (num_labels x 1).
        label_scores = np.expand_dims(self.W.dot(x_i), axis = 1)

        # One-hot encode true label (num_labels x 1).
        y_one_hot = np.zeros((np.size(self.W, 0),1))
        y_one_hot[y_i] = 1

        # Softmax function
        # This gives the label probabilities according to the model (num_labels x 1).
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
        
        # SGD update. W is num_labels x num_features.
        self.W = self.W + learning_rate * (y_one_hot - label_probabilities).dot(np.expand_dims(x_i, axis = 1).T)
        
        
        
class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
            
    def __init__(self, n_classes, n_features, hidden_size, n_layers=1):
        # Initialize an MLP with a single hidden layer.
        # n_classes: number of output classes
        # n_features: number of input features
        # hidden_size: number of hidden units
        
        # Initialize weights with normal distribution
        # Initialize weights with normal distribution
        self.n_layers = n_layers
        
        W1 = np.random.normal(loc=0.1, scale=0.1, size=(hidden_size, n_features))
        W2 = np.random.normal(loc=0.1, scale=0.1, size=(n_classes, hidden_size))
        
        self.W = [W1, W2]

        # Initialize biases with zero vectors
        b1 = np.zeros((hidden_size, 1))
        b2 = np.zeros((n_classes, 1))
        
        self.b = [b1, b2]
        
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_features = n_features


    def relu(self, x):
        res = np.array(x, copy=True)
        # replace every entry that is less than 0 with a 0
        res[res < 0] = 0
        return res

    def relu_prime(self, x):
        res = np.array(x, copy=True)
        res[res < 0] = 0
        res[res > 0] = 1
        return res

    def output(self, h):
        """
        h: a (n_class x n_points) matrix

        returns a vector of size n_points
        each entry corresponds to the label with highest "score"
        """
        return np.argmax(h, axis=0)


    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        h = X.T
        for i in range(self.n_layers):
            z = np.dot(self.W[i], h) + self.b[i]

            h = self.relu(z)

        h = np.dot(self.W[-1], h) + self.b[-1]

        return self.output(h)


    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


    def train_epoch(self, X, y, learning_rate=0.001):
        total_loss = 0
        for (x_i, y_i) in zip(X, y):
            # Compute the forward pass of the network. Save the values of hidden
            
            grad_z_l, hidden, g_primes = self.forward(x_i)
   
            loss = self.compute_loss(grad_z_l, y_i, X)
            total_loss+=loss
            
            grad_z_l[y_i] -= 1

            self.backward(x_i, y_i, grad_z_l, hidden, g_primes, learning_rate)
        print("len X", len(X))
        return [total_loss]  # Compute the mean of the loss values
        
        
    def forward(self, x):
        hiddens = []
        g_primes = []
        
        h = x.reshape((len(x), 1))
        hiddens.append(h)
        
        # compute hidden layers
        for i in range(self.n_layers):
            z = self.W[i].dot(h) + self.b[i]
            
            h = self.relu(z)
            hiddens.append(h)
            g_primes.append(self.relu_prime(z))
   
        h = np.dot(self.W[-1], h) + self.b[-1]
        
        output = utils.softmax(h)  
        
        return output, hiddens, g_primes
    
    
    def backward(self, x, y, grad_z_l, hiddens, g_primes, learning_rate):
        for l in range(self.n_layers, -1, -1):
            # Compute gradients of hidden layer parameters:
            grad_W_l = np.dot(grad_z_l, hiddens[l].T)
            grad_b_l = grad_z_l

            # Compute gradient of previous layer
            # only update if there is previous layer
            if l > 0:
                grad_h_l = np.dot(self.W[l].T, grad_z_l)
                grad_z_l = grad_h_l * g_primes[l-1]

            # Apply gradients to weights and bias
            self.W[l] -= learning_rate * grad_W_l
            self.b[l] -= learning_rate * grad_b_l
    
    
    def compute_loss(self, output, y, X):
        # compute loss
        loss = -np.sum(y * np.log(output + 1e-10)) / len(X)
                
        return loss    


def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()
    plt.savefig('images/q1_2b_accs.png')

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.show()
    plt.savefig('images/q1_2b_loss.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        print ("loss", loss)
        train_accs.append(model.evaluate(train_X, train_y))
        print("train_accs", train_accs)
        valid_accs.append(model.evaluate(dev_X, dev_y))
        print("valid_accs", valid_accs)
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss[-1], train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()