'''
This code trains the JODIE model for the given dataset. 
The task is: interaction prediction.

How to run: 
$ python jodie.py --network reddit --model jodie --epochs 50

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

import time

from library_data import *
import library_models as lib
from library_models import *

# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--network', required=True, help='Name of the network/dataset')
parser.add_argument('--model', default="jodie", help='Model name to save output in file')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train the model')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
parser.add_argument('--state_change', default=True, type=bool, help='True if training with state change of users along with interaction prediction. False otherwise. By default, set to True.')
args = parser.parse_args()

args.datapath = "data/%s.csv" % args.network
if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# LOAD DATA
[usertoid, user1_sequence_id, user1_timediffs_sequence, user1_previous_userid_sequence,
user2_sequence_id, user2_timediffs_sequence, user2_previous_userid_sequence,
 timestamp_sequence, feature_sequence, y_true] = load_network(args)
num_interactions = len(user1_sequence_id)
num_users = len(usertoid)
num_features = len(feature_sequence[0])
true_labels_ratio = len(y_true)/(1.0+sum(y_true)) # +1 in denominator in case there are no state change labels, which will throw an error. 
print("*** Network statistics:\n  %d users\n  %d interactions\n  ***\n\n" % (num_users, num_interactions))

# SET TRAINING, VALIDATION, TESTING, and TBATCH BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion) 
test_start_idx = int(num_interactions * (args.train_proportion+0.1))
test_end_idx = int(num_interactions * (args.train_proportion+0.2))

# SET BATCHING TIMESPAN
'''
Timespan is the frequency at which the batches are created and the JODIE model is trained. 
As the data arrives in a temporal order, the interactions within a timespan are added into batches (using the T-batch algorithm). 
The batches are then used to train JODIE. 
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
Longer timespan leads to less frequent model updates. 
'''
timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / 500 

# INITIALIZE MODEL AND PARAMETERS
model = JODIE(args, num_features, num_users)
MSELoss = nn.MSELoss()

# INITIALIZE EMBEDDING
initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim), dim=0))
# the initial user embeddings are learned during training as well
model.initial_user_embedding = initial_user_embedding

user_embeddings = initial_user_embedding.repeat(num_users, 1) # initialize all users to the same embedding 

# INITIALIZE MODEL
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# RUN THE JODIE MODEL
'''
THE MODEL IS TRAINED FOR SEVERAL EPOCHS. IN EACH EPOCH, JODIES USES THE TRAINING SET OF INTERACTIONS TO UPDATE ITS PARAMETERS.
'''
print("*** Training the JODIE model for %d epochs ***" % args.epochs)

# variables to help using tbatch cache between epochs
is_first_epoch = True
cached_tbatches_user1 = {}
cached_tbatches_user2 = {}
cached_tbatches_interactionids = {}
cached_tbatches_feature = {}
cached_tbatches_user1_timediffs = {}
cached_tbatches_user2_timediffs = {}
cached_tbatches_previous_user1 = {}
cached_tbatches_previous_user2 = {}

with trange(args.epochs) as progress_bar1:
    for ep in progress_bar1:
        progress_bar1.set_description('Epoch %d of %d' % (ep, args.epochs))

        epoch_start_time = time.time()

        optimizer.zero_grad()
        reinitialize_tbatches()
        total_loss, loss, total_interaction_count = 0, 0, 0

        tbatch_start_time = None
        tbatch_to_insert = -1
        tbatch_full = False

        # TRAIN TILL THE END OF TRAINING INTERACTION IDX
        with trange(train_end_idx) as progress_bar2:
            for j in progress_bar2:
                progress_bar2.set_description('Processed %dth interactions' % j)

                if is_first_epoch:
                    # READ INTERACTION J
                    user1id = user1_sequence_id[j]
                    user2id = user2_sequence_id[j]
                    feature = feature_sequence[j]
                    user1_timediff = user1_timediffs_sequence[j]
                    user2_timediff = user2_timediffs_sequence[j]

                    # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
                    tbatch_to_insert = max(lib.tbatchid_user[user1id], lib.tbatchid_user[user2id]) + 1
                    lib.tbatchid_user[user1id] = tbatch_to_insert
                    lib.tbatchid_user[user2id] = tbatch_to_insert

                    lib.current_tbatches_user1[tbatch_to_insert].append(user1id)
                    lib.current_tbatches_user2[tbatch_to_insert].append(user2id)
                    lib.current_tbatches_feature[tbatch_to_insert].append(feature)
                    lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
                    lib.current_tbatches_user1_timediffs[tbatch_to_insert].append(user1_timediff)
                    lib.current_tbatches_user2_timediffs[tbatch_to_insert].append(user2_timediff)
                    lib.current_tbatches_previous_user1[tbatch_to_insert].append(user1_previous_userid_sequence[j])
                    lib.current_tbatches_previous_user2[tbatch_to_insert].append(user2_previous_userid_sequence[j])

                timestamp = timestamp_sequence[j]
                if tbatch_start_time is None:
                    tbatch_start_time = timestamp

                # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES, FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
                if timestamp - tbatch_start_time > tbatch_timespan:
                    tbatch_start_time = timestamp # RESET START TIME FOR THE NEXT TBATCHES

                    # ITERATE OVER ALL T-BATCHES
                    if not is_first_epoch:
                        lib.current_tbatches_user1 = cached_tbatches_user1[timestamp]
                        lib.current_tbatches_user2 = cached_tbatches_user2[timestamp]
                        lib.current_tbatches_interactionids = cached_tbatches_interactionids[timestamp]
                        lib.current_tbatches_feature = cached_tbatches_feature[timestamp]
                        lib.current_tbatches_user1_timediffs = cached_tbatches_user1_timediffs[timestamp]
                        lib.current_tbatches_user2_timediffs = cached_tbatches_user2_timediffs[timestamp]
                        lib.current_tbatches_previous_user1 = cached_tbatches_previous_user1[timestamp]
                        lib.current_tbatches_previous_user2 = cached_tbatches_previous_user2[timestamp]


                    with trange(len(lib.current_tbatches_user1)) as progress_bar3:
                        for i in progress_bar3:
                            progress_bar3.set_description('Processed %d of %d T-batches ' % (i, len(lib.current_tbatches_user1)))
                            
                            total_interaction_count += len(lib.current_tbatches_interactionids[i])

                            # LOAD THE CURRENT TBATCH
                            if is_first_epoch:
                                lib.current_tbatches_user1[i] = torch.LongTensor(lib.current_tbatches_user1[i])
                                lib.current_tbatches_user2[i] = torch.LongTensor(lib.current_tbatches_user2[i])
                                lib.current_tbatches_interactionids[i] = torch.LongTensor(lib.current_tbatches_interactionids[i])
                                lib.current_tbatches_feature[i] = torch.Tensor(lib.current_tbatches_feature[i])

                                lib.current_tbatches_user1_timediffs[i] = torch.Tensor(lib.current_tbatches_user1_timediffs[i])
                                lib.current_tbatches_user2_timediffs[i] = torch.Tensor(lib.current_tbatches_user2_timediffs[i])
                                lib.current_tbatches_previous_user1[i] = torch.LongTensor(lib.current_tbatches_previous_user1[i])
                                lib.current_tbatches_previous_user2[i] = torch.LongTensor(
                                    lib.current_tbatches_previous_user2[i])

                            tbatch_user1ids = lib.current_tbatches_user1[i] # Recall "lib.current_tbatches_user[i]" has unique elements
                            tbatch_user2ids = lib.current_tbatches_user2[i] # Recall "lib.current_tbatches_item[i]" has unique elements
                            tbatch_interactionids = lib.current_tbatches_interactionids[i]
                            feature_tensor = Variable(lib.current_tbatches_feature[i]) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                            user1_timediffs_tensor = Variable(lib.current_tbatches_user1_timediffs[i]).unsqueeze(1)
                            user2_timediffs_tensor = Variable(lib.current_tbatches_user2_timediffs[i]).unsqueeze(1)
                            tbatch_user1ids_previous = lib.current_tbatches_previous_user1[i]
                            tbatch_user2ids_previous = lib.current_tbatches_previous_user2[i]
                            user1_embedding_previous = user_embeddings[tbatch_user1ids_previous, :]
                            user2_embedding_previous = user_embeddings[tbatch_user2ids_previous, :]

                            # PROJECT USER EMBEDDING TO CURRENT TIME
                            user1_embedding_input = user_embeddings[tbatch_user1ids,:]
                            user2_embedding_input = user_embeddings[tbatch_user2ids, :]
                            user1_projected_embedding = model.forward(user1_embedding_input, user1_embedding_previous,
                                                                      timediffs=user1_timediffs_tensor,
                                                                      features=feature_tensor, select='project')
                            user2_projected_embedding = model.forward(user2_embedding_input, user2_embedding_previous,
                                                                      timediffs=user2_timediffs_tensor,
                                                                      features=feature_tensor, select='project')

                            user1_previous_user_embedding = torch.cat(
                                [user1_projected_embedding, user1_embedding_previous], dim=1)

                            user2_previous_user_embedding = torch.cat(
                                [user2_projected_embedding, user2_embedding_previous], dim=1)

                            # PREDICT NEXT USER EMBEDDING
                            predicted_user1_next_user_embedding = model.predict_user_embedding(user1_previous_user_embedding)
                            predicted_user2_next_user_embedding = model.predict_user_embedding(user2_previous_user_embedding)

                            # CALCULATE PREDICTION LOSS
                            loss += MSELoss(predicted_user1_next_user_embedding, user2_embedding_input.detach())

                            loss += MSELoss(predicted_user2_next_user_embedding, user1_embedding_input.detach())

                            # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                            user1_embedding_output = model.forward(user1_embedding_input, user2_embedding_input,
                                                                   timediffs=user1_timediffs_tensor,
                                                                   features=feature_tensor, select='user_update')
                            user2_embedding_output = model.forward(user2_embedding_input, user1_embedding_input,
                                                                   timediffs=user2_timediffs_tensor,
                                                                   features=feature_tensor, select='user_update')
                            user_embeddings[tbatch_user1ids,:] = user1_embedding_output
                            user_embeddings[tbatch_user2ids, :] = user2_embedding_output

                            # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                            loss += MSELoss(user1_embedding_output, user1_embedding_input.detach())
                            loss += MSELoss(user2_embedding_output, user2_embedding_input.detach())


                    # BACKPROPAGATE ERROR AFTER END OF T-BATCH
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # RESET LOSS FOR NEXT T-BATCH
                    loss = 0
                    # Detachment is needed to prevent double propagation of gradient
                    user_embeddings.detach_()
                   
                    # REINITIALIZE
                    if is_first_epoch:
                        cached_tbatches_user1[timestamp] = lib.current_tbatches_user1
                        cached_tbatches_user2[timestamp] = lib.current_tbatches_user2
                        cached_tbatches_interactionids[timestamp] = lib.current_tbatches_interactionids
                        cached_tbatches_feature[timestamp] = lib.current_tbatches_feature
                        cached_tbatches_user1_timediffs[timestamp] = lib.current_tbatches_user1_timediffs
                        cached_tbatches_user2_timediffs[timestamp] = lib.current_tbatches_user2_timediffs
                        cached_tbatches_previous_user1[timestamp] = lib.current_tbatches_previous_user1
                        cached_tbatches_previous_user2[timestamp] = lib.current_tbatches_previous_user2
                        
                        reinitialize_tbatches()
                        tbatch_to_insert = -1

        is_first_epoch = False # as first epoch ends here
        print("Last epoch took {} minutes".format((time.time()-epoch_start_time)/60))
        # END OF ONE EPOCH 
        print("\n\nTotal loss in this epoch = %f" % (total_loss))
        user_embeddings_save = copy.deepcopy(user_embeddings)
        # SAVE CURRENT MODEL TO DISK TO BE USED IN EVALUATION.
        save_model(model, optimizer, args, ep, user_embeddings_save, train_end_idx)

        user_embeddings = initial_user_embedding.repeat(num_users, 1)

# END OF ALL EPOCHS. SAVE FINAL MODEL DISK TO BE USED IN EVALUATION.
print("\n\n*** Training complete. Saving final model. ***\n\n")
save_model(model, optimizer, args, ep, user_embeddings_save, train_end_idx)

