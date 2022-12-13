'''
This is a supporting library for the loading the data.

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

from __future__ import division
import numpy as np
import random
import sys
import operator
import copy
from collections import defaultdict
import os, re
import argparse
from sklearn.preprocessing import scale

# LOAD THE NETWORK
def load_network(args, time_scaling=True):
    '''
    This function loads the input network.

    The network should be in the following format:
    One line per interaction/edge.
    Each line should be: user, item, timestamp, state label, array of features.
    Timestamp should be in cardinal format (not in datetime).
    State label should be 1 whenever the user state changes, 0 otherwise. If there are no state labels, use 0 for all interactions.
    Feature list can be as long as desired. It should be atleast 1 dimensional. If there are no features, use 0 for all interactions. 
    '''

    network = args.network
    datapath = args.datapath

    user_sequence = []
    item_sequence = []
    label_sequence = []
    feature_sequence = []
    timestamp_sequence = []
    start_timestamp = None
    y_true_labels = []

    print("\n\n**** Loading %s network from file: %s ****" % (network, datapath))
    f = open(datapath,"r")
    f.readline()
    for cnt, l in enumerate(f):
        # FORMAT: user, item, timestamp, state label, feature list 
        ls = l.strip().split(",")
        user_sequence.append(ls[0])
        item_sequence.append(ls[1])
        if start_timestamp is None:
            start_timestamp = float(ls[2])
        timestamp_sequence.append(float(ls[2]) - start_timestamp) 
        y_true_labels.append(int(ls[3])) # label = 1 at state change, 0 otherwise
        feature_sequence.append(list(map(float,ls[4:])))
    f.close()

    user_sequence = np.array(user_sequence)
    item_sequence = np.array(item_sequence)
    timestamp_sequence = np.array(timestamp_sequence)

    print("Formatting user sequence")
    nodeid = 0
    usertoid = {}
    user1_timedifference_sequence = []
    user2_timedifference_sequence = []
    user_current_timestamp = defaultdict(float)
    user1_previous_userid_sequence = []
    user2_previous_userid_sequence = []
    user_latest_userid = defaultdict(int)  # lambda: num_users
    for cnt in range(len(user_sequence)):
        user1 = user_sequence[cnt]
        user2 = item_sequence[cnt]
        if user1 not in usertoid:
            usertoid[user1] = nodeid
            nodeid += 1
        if user2 not in usertoid:
            usertoid[user2] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        user1_timedifference_sequence.append(timestamp - user_current_timestamp[user1])
        user2_timedifference_sequence.append(timestamp - user_current_timestamp[user2])
        user_current_timestamp[user1] = timestamp
        user_current_timestamp[user2] = timestamp
        user1_previous_userid_sequence.append(user_latest_userid[user1])
        user2_previous_userid_sequence.append(user_latest_userid[user2])
        user_latest_userid[user1] = usertoid[user2]
        user_latest_userid[user2] = usertoid[user1]
    # num_users = len(user2id)
    user1_sequence_id = [usertoid[user] for user in user_sequence]
    user2_sequence_id = [usertoid[user] for user in item_sequence]

    if time_scaling:
        print("Scaling timestamps")
        user1_timedifference_sequence = scale(np.array(user1_timedifference_sequence) + 1)
        user2_timedifference_sequence = scale(np.array(user2_timedifference_sequence) + 1)

    print("*** Network loading completed ***\n\n")
    return [usertoid, user1_sequence_id, user1_timedifference_sequence, user1_previous_userid_sequence, user2_sequence_id,
            user2_timedifference_sequence, user2_previous_userid_sequence, timestamp_sequence, feature_sequence, y_true_labels]

