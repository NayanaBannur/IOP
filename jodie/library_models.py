'''
This is a supporting library with the code of the model.

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

from __future__ import division
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import sys
from collections import defaultdict
import os
import gpustat
from itertools import chain
from tqdm import tqdm, trange, tqdm_notebook, tnrange
import csv
import json

PATH = "./"

total_reinitialization_count = 0

# A NORMALIZATION LAYER
class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)


# THE JODIE MODULE
class JODIE(nn.Module):
    def __init__(self, args, num_features, num_users):
        super(JODIE,self).__init__()

        print("*** Initializing the JODIE model ***")
        self.modelname = args.model
        self.embedding_dim = args.embedding_dim

        print("Initializing user embeddings")
        self.initial_user_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))

        rnn_input_size_users = self.embedding_dim + 1 + num_features

        print("Initializing user RNNs")
        self.user_rnn = nn.RNNCell(rnn_input_size_users, self.embedding_dim)

        print("Initializing linear layers")
        self.linear_layer1 = nn.Linear(self.embedding_dim, 50)
        self.linear_layer2 = nn.Linear(50, 2)
        self.prediction_layer = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.embedding_layer = NormalLinear(1, self.embedding_dim)
        print("*** JODIE initialization complete ***\n\n")
        
    def forward(self, user1_embeddings, user2_embeddings, timediffs=None, features=None, select=None):
        if select == 'user_update':
            input2 = torch.cat([user2_embeddings, timediffs, features], dim=1)
            user_embedding_output = self.user_rnn(input2, user1_embeddings)
            return F.normalize(user_embedding_output)

        elif select == 'project':
            user_projected_embedding = self.context_convert(user1_embeddings, timediffs, features)
            return user_projected_embedding

    def context_convert(self, embeddings, timediffs, features):
        new_embeddings = embeddings * (1 + self.embedding_layer(timediffs))
        return new_embeddings

    def predict_user_embedding(self, user_embeddings):
        X_out = self.prediction_layer(user_embeddings)
        return X_out


# INITIALIZE T-BATCH VARIABLES
def reinitialize_tbatches():
    global current_tbatches_interactionids, current_tbatches_user1, current_tbatches_user2, \
        current_tbatches_timestamp, current_tbatches_feature, current_tbatches_label, current_tbatches_previous_user1, \
        current_tbatches_previous_user2
    global tbatchid_user, current_tbatches_user1_timediffs, current_tbatches_user2_timediffs, \
        current_tbatches_user1_timediffs_next, current_tbatches_user2_timediffs_next

    # list of users of each tbatch up to now
    current_tbatches_interactionids = defaultdict(list)
    current_tbatches_user1 = defaultdict(list)
    current_tbatches_user2 = defaultdict(list)
    current_tbatches_timestamp = defaultdict(list)
    current_tbatches_feature = defaultdict(list)
    current_tbatches_label = defaultdict(list)
    current_tbatches_previous_user1 = defaultdict(list)
    current_tbatches_previous_user2 = defaultdict(list)
    current_tbatches_user1_timediffs = defaultdict(list)
    current_tbatches_user2_timediffs = defaultdict(list)
    current_tbatches_user1_timediffs_next = defaultdict(list)
    current_tbatches_user2_timediffs_next = defaultdict(list)

    # the latest tbatch a user is in
    tbatchid_user = defaultdict(lambda: -1)

    global total_reinitialization_count
    total_reinitialization_count +=1


# CALCULATE LOSS FOR THE PREDICTED USER STATE 
def calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_time_series, y_true, loss_function):
    # PREDICT THE LABEL FROM THE USER DYNAMIC EMBEDDINGS
    prob = model.predict_label(user_embeddings_time_series[tbatch_interactionids,:])
    y = Variable(torch.LongTensor(y_true)[tbatch_interactionids])
    
    loss = loss_function(prob, y)

    return loss


# SAVE TRAINED MODEL TO DISK
def save_model(model, optimizer, args, epoch, user_embeddings, train_end_idx, path=PATH):
    print("*** Saving embeddings and model ***")
    directory = os.path.join(path, 'saved_models/%s/%d' % (args.network, epoch))
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(model, os.path.join(directory, 'model.pt'))
    torch.save(optimizer, os.path.join(directory, 'optimizer.pt'))
    np.save(os.path.join(directory, 'user_embeddings'), user_embeddings.data.cpu().numpy())
    print("*** Saved embeddings and model to file ***\n\n") # %s % filename


# # LOAD PREVIOUSLY TRAINED AND SAVED MODEL
# def load_model(model, optimizer, args, epoch):
#     modelname = args.model
#     filename = PATH + "saved_models/%s/checkpoint.%s.ep%d.tp%.1f.pth.tar" % (args.network, modelname, epoch, args.train_proportion)
#     checkpoint = torch.load(filename)
#     print("Loading saved embeddings and model: %s" % filename)
#     args.start_epoch = checkpoint['epoch']
#     user_embeddings = Variable(torch.from_numpy(checkpoint['user_embeddings']))
#     try:
#         train_end_idx = checkpoint['train_end_idx']
#     except KeyError:
#         train_end_idx = None
#
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#
#     return [model, optimizer, user_embeddings, train_end_idx]

# LOAD PREVIOUSLY TRAINED AND SAVED MODEL
def load_model(model, optimizer, args, epoch, path=PATH):
    directory = os.path.join(path, 'saved_models/%s/%d' % (args.network, epoch))
    model = torch.load(os.path.join(directory, 'model.pt'))
    optimizer = torch.load(os.path.join(directory, 'optimizer.pt'))
    ue = np.load(os.path.join(directory, 'user_embeddings.npy'))
    user_embeddings = Variable(torch.from_numpy(ue))
    return [model, optimizer, user_embeddings]


# SET USER EMBEDDINGS TO THE END OF THE TRAINING PERIOD
def set_embeddings_training_end(user_embeddings, user_embeddings_time_series, user_data_id, train_end_idx):
    userid2lastidx = {}
    for cnt, userid in enumerate(user_data_id[:train_end_idx]):
        userid2lastidx[userid] = cnt

    try:
        embedding_dim = user_embeddings_time_series.size(1)
    except:
        embedding_dim = user_embeddings_time_series.shape[1]
    for userid in userid2lastidx:
        user_embeddings[userid, :embedding_dim] = user_embeddings_time_series[userid2lastidx[userid]]

    user_embeddings.detach_()
