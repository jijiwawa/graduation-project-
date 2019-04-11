import numpy as np

import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.regularizers import l2, activity_l2
from keras.models import Sequential, Graph, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout,dot
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import sys
import argparse
import multiprocessing as mp
from scipy._lib.six import xrange
import scipy.sparse as sp
import heapq
import numpy as np
import time
from prepare_datasets.Recommendation import Recommendation

class NurtualBaseRecommendation(Recommendation):

    def __init__(self, file):
        super(NurtualBaseRecommendation, self).__init__(file)

    def get_model(self,W_user=[20, 10],W_item=[20,10]):
        assert len(layers) == len(reg_layers)
        num_layer = len(layers)  # Number of layers
        # Input variables
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        # MLP layers
        for idx in xrange(1, num_layer):
            layer_user = Dense(W_user[idx], activation='relu', name='layer%d' % idx)
            layer_item = Dense(W_item[idx], activation='relu', name='layer%d' % idx)
            user_input = layer_user(user_input)
            item_input = layer_item(item_input)

        # Final prediction
        prediction = dot([user_input,item_input])/(len(user_input)*len(item_input))
        model = Model(input=[user_input, item_input], output=prediction)
        return model


    def get_train_instances(self,train, neg_ratio):
        user_input, item_input, labels = [], [], []
        num_users = train.shape[0]
        num_items = train.shape[1]
        for (u, i) in train.keys():
            # positive instance
            user_vector_u = sp.dok_matrix.toarray(train)[u]
            user_vector_i = sp.dok_matrix.transpose(train).toarray()[i]
            user_input.append(user_vector_u)
            item_input.append(user_vector_i)
            labels.append(train[u,i])
            # negative instances
            for t in xrange(neg_ratio):
                j = np.random.randint(num_items)
                while train.has_key((u, j)):
                    j = np.random.randint(num_items)
                user_input.append(user_vector_u)
                user_vector_j = sp.dok_matrix.transpose(train).toarray()[j]
                item_input.append(user_vector_j)
                labels.append(0)
        return user_input, item_input, labels

def get_model(self,W_user=[20, 10],W_item=[20,10]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # MLP layers
    for idx in xrange(1, num_layer):
        layer_user = Dense(W_user[idx], activation='relu', name='layer%d' % idx)
        layer_item = Dense(W_item[idx], activation='relu', name='layer%d' % idx)
        user_input = layer_user(user_input)
        item_input = layer_item(item_input)

    # Final prediction
    prediction = dot([user_input,item_input])/(len(user_input)*len(item_input))
    model = Model(input=[user_input, item_input], output=prediction)
    return model


def get_train_instances(self,train, neg_ratio):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    num_items = train.shape[1]
    for (u, i) in train.keys():
        # positive instance
        user_vector_u = sp.dok_matrix.toarray(train)[u]
        user_vector_i = sp.dok_matrix.transpose(train).toarray()[i]
        user_input.append(user_vector_u)
        item_input.append(user_vector_i)
        labels.append(train[u,i])
        # negative instances
        for t in xrange(neg_ratio):
            j = np.random.randint(num_items)
            while train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(user_vector_u)
            user_vector_j = sp.dok_matrix.transpose(train).toarray()[j]
            item_input.append(user_vector_j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    batch_size = 256
    topK = 10
    csv_path = 'C:\\Users\\suxik\\Desktop\\text\\graduation-project-\\prepare_datasets\\test.csv'
    csv_path1 = 'C:\\Users\\suxik\Desktop\\text\graduation-project-\\prepare_datasets\\ml-1m.train.rating'
    # Loading data
    t1 = time()
    Rec = Recommendation(csv_path)
    train, testRatings, testNegatives = Rec.ratingMatrix, Rec.testRatings, Rec.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    model = get_model(num_users, num_items, layers, reg_layers)
    model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

    # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))

    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in xrange(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)

        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best MLP model is saved to %s" % (model_out_file))
