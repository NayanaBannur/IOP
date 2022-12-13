'''
This code evaluates the validation and test performance in an epoch of the model trained in jodie.py.
The task is: interaction prediction, i.e., predicting which item will a user interact with?

To calculate the performance for one epoch:
$ python evaluate_interaction_prediction.py --network reddit --model jodie --epoch 49

To calculate the performance for all epochs, use the bash file, evaluate_all_epochs.sh, which calls this file once for every epoch.

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019.
'''

from library_data import *
from library_models import *
from utils import *
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--network', required=True, help='Network name')
parser.add_argument('--model', default='jodie', help="Model name")
parser.add_argument('--epoch', default=0, type=int, help='Epoch id to load')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Proportion of training interactions')
parser.add_argument('--set_indice_length', default=3, type=int)
args = parser.parse_args()
args.datapath = "data/%s.csv" % args.network

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# LOAD NETWORK
[usertoid, user1_sequence_id, user1_timediffs_sequence, _, user2_sequence_id, user2_timediffs_sequence, _,
 timestamp_sequence, _, _] = load_network(args)
num_interactions = len(user1_sequence_id)
num_users = len(usertoid)
print("*** Network statistics:\n  %d users\n  %d interactions\n***\n\n" % (num_users, num_interactions))


# SET TRAIN, VALIDATION, AND TEST BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion)
test_start_idx = int(num_interactions * (args.train_proportion + 0.1))
test_end_idx = int(num_interactions * (args.train_proportion + 0.2))
set_indice_length = args.set_indice_length
embedding_dim = args.embedding_dim
n_classes = math.factorial(int((set_indice_length * (set_indice_length - 1)) / 2))

# INITIALIZE MODEL PARAMETERS
temp_model = JODIE(args, 1, num_users)

# INITIALIZE MODEL
learning_rate = 1e-3
temp_optimizer = optim.Adam(temp_model.parameters(), lr=learning_rate, weight_decay=1e-5)

# LOAD THE MODEL
temp_model, temp_optimizer, user_embeddings = load_model(temp_model, temp_optimizer, args, args.epoch)

# DATA
train_users = user1_sequence_id[:validation_start_idx]
train_users.extend(user2_sequence_id[:validation_start_idx])
train_users = set(train_users)

train_edges = np.array([user1_sequence_id[:validation_start_idx], user2_sequence_id[:validation_start_idx]]).T.tolist()
test_edges = np.array([user1_sequence_id[validation_start_idx:], user2_sequence_id[validation_start_idx:]]).T
mask = np.isin(test_edges[:, 0], list(train_users))
mask &= np.isin(test_edges[:, 1], list(train_users))
test_edges = test_edges[mask].tolist()

train_ts = timestamp_sequence[:validation_start_idx]
test_ts = timestamp_sequence[validation_start_idx:]

train_G, train_set_indices, train_labels = get_data(train_edges, train_ts, set_indice_length)
test_G, test_set_indices, test_labels = get_data(test_edges, test_ts, set_indice_length)

# MODEL
hidden_dim = embedding_dim
model = nn.Sequential(
            nn.Linear(set_indice_length * embedding_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, n_classes)
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
loss_fn = torch.nn.functional.cross_entropy

n_train_val = len(train_set_indices)
n_val = int(0.1 * n_train_val)
n_train = n_train_val - n_val

print(n_train, n_val)

train_dataset = JodieDataset(set_indice_length, train_set_indices[:n_train], train_labels[:n_train], user_embeddings)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = JodieDataset(set_indice_length, train_set_indices[n_train:], train_labels[n_train:], user_embeddings)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataset = JodieDataset(set_indice_length, test_set_indices, test_labels, user_embeddings)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def metrics(predictions, labels):
    pred_class = np.argmax(predictions, axis=1)
    acc = (pred_class == labels).sum() / len(labels)
    bleu = bleu_metric(predictions, labels.tolist())
    meteor = meteor_metric(predictions, labels.tolist())
    f1 = f1_score(pred_class, labels, average='macro')
    kt = kendall_tau_metric(predictions, labels)
    sp = spearman_metric(predictions, labels)
    return {'acc': acc, 'bleu': bleu, 'meteor': meteor, 'kt_corr': kt.correlation, 'kt_pval': kt.pvalue,
            'sp_corr': sp.correlation, 'sp_pval': sp.pvalue, 'f1': f1}


all_results = []
n_test_list = [10, 20, 50, 100, 150, 200, 300, 500]

for run in range(10):
    print('Run ', run)

    run_results = []
    for epoch in tqdm(range(50)):

        metrics_dict = {}

        train_loss = 0.0
        preds = []
        labels = []
        for c, batch in enumerate(train_dataloader):
            x = batch['x']
            label = batch['label']
            optimizer.zero_grad()
            pred = model(x)

            preds.append(pred.detach().numpy())
            labels.append(label.detach().numpy())

            loss = loss_fn(pred, label)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        preds = np.vstack(preds)
        labels = np.concatenate(labels)
        train_loss /= len(labels)

        train_metrics = metrics(preds, labels)

        preds = []
        labels = []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                x = batch['x']
                label = batch['label']
                pred = model(x)

                preds.append(pred.detach().numpy())
                labels.append(label.detach().numpy())

                loss = loss_fn(pred, label)

                val_loss += loss.item()

            preds = np.vstack(preds)
            labels = np.concatenate(labels)
            val_loss /= len(labels)

            val_metrics = metrics(preds, labels)
            val_metrics = {'val' + '_' + k: v for k, v in val_metrics.items()}

        metrics_dict.update(val_metrics)

        preds = []
        labels = []
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_dataloader:
                x = batch['x']
                label = batch['label']
                pred = model(x)

                preds.append(pred.detach().numpy())
                labels.append(label.detach().numpy())

                loss = loss_fn(pred, label)

                test_loss += loss.item()

            preds = np.vstack(preds)
            labels = np.concatenate(labels)
            test_loss /= len(labels)

            test_metrics = {}
            for n_test in n_test_list:
                m = metrics(preds[:n_test, :], labels[:n_test])
                test_metrics.update({f'test_{n_test}_{k}': v for k, v in m.items()})

        metrics_dict.update(test_metrics)

        run_results.append(metrics_dict)

    all_metrics = pd.DataFrame(run_results)
    print(all_metrics)

    best_metrics = {}
    for key in ['acc', 'bleu', 'meteor', 'f1']:
        idx = np.argpartition(-all_metrics[f"val_{key}"], 1)[0]
        for n_test in n_test_list:
            best_metrics[f'{key}_{n_test}'] = all_metrics[f"test_{n_test}_{key}"][idx]
    for key in ['kt', 'sp']:
        for n_test in n_test_list:
            idx = np.argpartition(-all_metrics[f"val_{key}_corr"], 1)[0]
            best_metrics[f'{key}_{n_test}_corr'] = all_metrics[f"test_{n_test}_{key}_corr"][idx]
            best_metrics[f'{key}_{n_test}_pval'] = all_metrics[f"test_{n_test}_{key}_pval"][idx]
    best_metrics['run'] = run
    print(best_metrics)
    all_results.append(best_metrics)

all_results = pd.DataFrame(all_results)
all_results.to_csv(f'./outputs/results_dyn_emb_test_sizes.csv', index=False)
