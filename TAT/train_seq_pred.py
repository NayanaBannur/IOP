# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# Experiments: n-3, n-4, concat/mean, decoder, attn decoder, gru layers, hidden size?

import warnings

import torch
import numpy as np
import copy
import pandas as pd
import os
import scipy
import scipy.stats
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from torch import Tensor
from tqdm import tqdm
from collections import OrderedDict, namedtuple
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from typing import List, Dict
import math
from itertools import combinations, permutations

criterion = torch.nn.CrossEntropyLoss()
CorrCoeff = namedtuple('corr_coeff', ['correlation', 'pvalue'])

def get_device(gpu_index):
    if gpu_index >= 0:
        assert torch.cuda.is_available(), 'cuda not available'
        assert gpu_index >= 0 and gpu_index < torch.cuda.device_count(), 'gpu index out of range'
        return torch.device('cuda:{}'.format(gpu_index))
    else:
        return torch.device('cpu')

def get_optimizer(model, args):
    optim = args.optimizer
    if optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif optim == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise NotImplementedError


def train_model(model, dataloaders, args, logger):
    device = get_device(args.gpu)
    model = model.to(device)
    target_model = copy.deepcopy(model)
    target_model = target_model.to(device)

    train_loader, val_loader, test_loader = dataloaders
    optimizer = get_optimizer(model, args)
    metrics = {'loss': loss_metric,
               'acc': acc_metric,
               # 'auc': roc_auc_metric, 'macro_f1': f1_score_metric, 'ranked_acc': ranked_acc_metric,
               'bleu': bleu_metric,
               'kendall_tau': kendall_tau_metric}

    recorder = Recorder(minmax={'loss': 0, 'acc': 1, 'auc': 1, 'macro_f1': 1, 'ranked_acc': 1, 'bleu': 1, 'kendall_tau': 1},
                        checkpoint_dir=args.checkpoint_dir, time_str=args.time_str)
    log_dir = Path(args.log_dir)/'tb_logs'/args.dataset/'_'.join([args.time_str, args.desc.replace(' ', '_')]) # add desc after time, as log dir name
    # summary_writer = TensorboardSummarizer(log_dir, hparams_dict=vars(args), desc=args.desc)
    summary_writer = None
    for step in range(args.epoch):
        optimize_model(model, target_model, train_loader, optimizer, logger, summary_writer, step,
                       perm_loss=args.perm_loss, set_indice_length=args.set_indice_length, prevent_repeat=args.prevent_repeat)
        train_metrics_results = eval_model(model, train_loader, metrics, desc='train_eval', start_step=step*len(train_loader),
                                           set_indice_length=args.set_indice_length, prevent_repeat=args.prevent_repeat)
        val_metrics_results = eval_model(model, val_loader, metrics, desc='val_eval', start_step=step*len(val_loader),
                                         recorder=recorder, step=step,
                                         set_indice_length=args.set_indice_length, prevent_repeat=args.prevent_repeat)
        test_metrics_results = eval_model(model, test_loader, metrics, desc='test_eval',
                                          start_step=step*len(test_loader), recorder=recorder, step=step,
                                          set_indice_length=args.set_indice_length, prevent_repeat=args.prevent_repeat)

        recorder.append_full_metrics(train_metrics_results, 'train')
        recorder.append_full_metrics(val_metrics_results, 'val')
        recorder.append_full_metrics(test_metrics_results, 'test')
        recorder.append_model_state(model.state_dict(), metrics)

        test_best_metric, _ = recorder.get_best_metric('test')
        train_latest_metric = recorder.get_latest_metric('train')
        val_latest_metric = recorder.get_latest_metric('val')
        test_latest_metric = recorder.get_latest_metric('test')

        print("Epoch: ", step)
        print("Train: ", train_latest_metric)
        print("Val: ", val_latest_metric)
        print("Test", test_latest_metric)

        recorder.save()
        recorder.save_model()

    # summary_writer.close()
    # overall statistics
    train_best_metric, train_best_epoch = recorder.get_best_metric('train')
    val_best_metric, val_best_epoch = recorder.get_best_metric('val')
    test_best_metric, test_best_epoch = recorder.get_best_metric('test')
    print("Train: ", train_best_metric)
    print("Val: ", val_best_metric)
    print("Test", test_best_metric)
    recorder.save()
    recorder.save_model()
    logger.info(f'record saved at {recorder.checkpoint_dir/recorder.time_str}')
    return recorder


def each_class_acc(predictions, labels):
    num_samples = np.zeros((5,), )
    num_correct = np.zeros((5,), )
    predictions = torch.argmax(predictions, dim=1).cpu().numpy()
    for i, (p, l) in enumerate(zip(predictions, labels)):
        num_samples[l] += 1
        if p == l:
            num_correct[l] += 1
    
    class_acc = num_correct / num_samples
    print(class_acc)

def model_device(model):
    """ return device of a model
    """
    return next(model.parameters()).device


def eval_model(model, dataloader, metrics, **kwargs):
    device = model_device(model)
    model.eval()
    predictions = []
    labels = []
    desc = kwargs.get('desc', 'desc')
    recorder = kwargs.get('recorder', None)
    epoch = str(kwargs.get('step', ''))
    l = int((kwargs["set_indice_length"] * (kwargs["set_indice_length"] - 1)) / 2)
    if recorder is not None:
        path = recorder.checkpoint_dir / recorder.time_str
    prevent_repeat = kwargs["prevent_repeat"]

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=desc):
            try:
                batch = batch.to(device)
                encoding = model(batch).reshape(1, len(batch), -1)
                decoder_input = torch.zeros(len(batch), dtype=torch.float32).reshape((-1, 1, 1))
                outputs = []
                for di in range(l):
                    decoder_output, decoder_hidden = model.decoder(decoder_input, encoding)
                    if prevent_repeat:
                        if len(outputs) > 0:
                            current_outputs = torch.cat(outputs, dim=2)
                            current_outputs = torch.cat([torch.zeros(len(batch), dtype=torch.int64).reshape((-1, 1, 1)),
                                                         current_outputs],
                                                        dim=2)
                        else:
                            current_outputs = torch.zeros(len(batch), dtype=torch.int64).reshape((-1, 1, 1))
                        decoder_output.scatter_(2, current_outputs, -float("inf"))
                    topv, topi = decoder_output.topk(1, dim=2)
                    decoder_input = topi.detach().to(torch.float32)
                    outputs.append(topi)

                outputs = torch.cat(outputs, dim=2).squeeze()
                predictions.append(outputs)
                labels.append(batch.y_seq)
            except RuntimeError:
                continue

    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)

    name = desc.split('_')[0]
    if recorder is not None:
        torch.save(predictions, path / f'{name}_predictions_{epoch}.pt')
        torch.save(labels, path / f'{name}_labels_{epoch}.pt')
    metrics_results = compute_metric(predictions, labels, metrics)
    return metrics_results


def acc_metric(predictions, labels):
    acc = (predictions == labels).all(dim=1).to(torch.float32).sum() / labels.shape[0]
    return acc.item()

def roc_auc_metric(predictions, labels):
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    predictions = scipy.special.softmax(predictions, axis=1)
    multi_class = 'ovr'
    if predictions.shape[1] == 2:
        predictions = predictions[:, 1]
        multi_class = 'raise'
    try:
        auc = roc_auc_score(labels, predictions, multi_class=multi_class)
    except ValueError:
        auc = 0
    return auc

def f1_score_metric(predictions, labels):
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    predictions = np.argmax(predictions, axis=1)
    macro_f1_score = f1_score(predictions, labels, average='macro')
    return macro_f1_score

def ranked_acc_metric(predictions, labels, rank=2):
    """ ACC@2
    """
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    class_rank = np.argsort(-1*predictions, axis=-1)
    labels = np.repeat(labels.reshape((-1, 1)), rank, axis=1)
    correct_predictions = np.max(class_rank[:, :rank] == labels, axis=1).flatten()
    acc = correct_predictions.sum()/labels.shape[0]
    return acc

def kendall_tau_metric(predictions, labels):
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    tau = 0.0
    pval = 0.0
    for i in range(predictions.shape[0]):
        kendall_tau = scipy.stats.kendalltau(predictions[i], labels[i])
        pred_tau = kendall_tau.correlation if not np.isnan(kendall_tau.correlation) else 0.0
        pred_pval = kendall_tau.pvalue if not np.isnan(kendall_tau.pvalue) else 0.0
        tau += pred_tau
        pval += pred_pval
    tau /= len(labels)
    pval /= len(labels)
    return CorrCoeff(tau, pval)


def spearman_metric(predictions, labels):
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    tau = 0.0
    pval = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(predictions.shape[0]):
            metric_val = scipy.stats.spearmanr(predictions[i], labels[i])
            pred_tau = metric_val.correlation if not np.isnan(metric_val.correlation) else 0.0
            pred_pval = metric_val.pvalue if not np.isnan(metric_val.pvalue) else 0.0
            tau += pred_tau
            pval += pred_pval
        tau /= len(labels)
        pval /= len(labels)
    return CorrCoeff(tau, pval)


def loss_metric(predictions, labels):
    """ cross entropy loss """
    if not isinstance(predictions, Tensor):
        predictions = torch.Tensor(predictions).cpu()
    if not isinstance(labels, Tensor):
        labels = torch.LongTensor(labels)
        labels = labels.to(predictions.device)

    loss = 0
    with torch.no_grad():
        for i in range(len(labels)):
            loss += torch.nn.functional.cross_entropy(predictions[i].to(torch.float32),
                                                      labels[i].to(torch.float32)).item()
    return loss / len(labels)


def label_to_order(label, l):
    for i, p in enumerate(permutations(range(l))):
        if i == label:
            return p


def fact_to_num(n):
    return {6: 3, 720: 6}[n]

def bleu_metric(predictions, labels, ngram=3):
    if ngram == 3:
        weights = (1 / 3, 1 / 3, 1 / 3, 0)
    elif ngram == 2:
        weights = (1 / 2, 1 / 2, 0, 0)
    else:
        raise NotImplementedError
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(predictions, Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, Tensor):
            labels = labels.cpu().tolist()
        bleu = 0
        for i in range(len(labels)):
            bleu += sentence_bleu([labels[i]], predictions[i], weights=weights)
        bleu /= len(labels)
        return bleu


def meteor_metric(predictions, labels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(predictions, Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, Tensor):
            labels = labels.cpu().tolist()
        meteor = 0
        for i in range(len(labels)):
            label = [[str(s) for s in labels[i]]]
            pred = [str(s) for s in predictions[i].tolist()]
            meteor += meteor_score(label, pred)
        meteor /= len(labels)
        return meteor

    
def compute_metric(predictions, labels, metrics):
    metrics_results = {}
    for key, f in metrics.items():
        metrics_results[key] = f(predictions, labels)    
    return metrics_results

def permute_batch(batch, permute_idx, permute_rand):
    
    device = batch.set_indice.device
    # index bases
    set_indice, batch_idx, num_graphs = batch.set_indice, batch.batch, batch.num_graphs
    set_indice_length = set_indice.shape[1]
    num_nodes = torch.eye(num_graphs)[batch_idx].to(device).sum(dim=0)
    zero = torch.tensor([0], dtype=torch.long).to(device)
    index_bases = torch.cat([zero, torch.cumsum(num_nodes, dim=0, dtype=torch.long)[:-1] ])
    # reset x
    for i in range(num_graphs):
        for j in range(set_indice_length):
            batch.x[index_bases[i] + set_indice[i][j]][:set_indice_length] = 0
    
    # permute set_indice
    permute_idx = torch.LongTensor(permute_idx).to(device)
    for i in range(num_graphs):
        batch.set_indice[i] = batch.set_indice[i][ permute_idx[permute_rand[i]]]
    
    # renew x
    new_set_indice = batch.set_indice
    for i in range(num_graphs):
        for j in range(set_indice_length):
            batch.x[index_bases[i] + new_set_indice[i][j]][j] = 1

    return batch

def determine_triad_class_(set_indice, timestamp):
    t = []
    set_indice_length = len(set_indice)
    for (i, j) in combinations(range(set_indice_length), 2):
        t.append(timestamp[(set_indice[i], set_indice[j])])
    times = list(sorted(t))
    times = np.array(times).reshape((1, -1))

    perm = np.array(list(permutations(t)))
    index = np.argmax(np.all(perm == times, axis=1))
    return index

def time_dict(set_indice, y):
    set_indice_length = len(set_indice)
    assert np.all(set_indice == np.array(list(range(1, set_indice_length + 1)), dtype=np.int))
    times = {}

    pairs = []
    for (i, j) in combinations(set_indice, 2):
        pairs.append((i, j))
    length = int(set_indice_length * (set_indice_length - 1) / 2)
    perm = permutations(range(length))

    for i, p in enumerate(perm):
        if y == i:
            for j in range(length):
                times[pairs[p[j]]] = j
            break

    for k in list(times.keys()):
        times[(k[1], k[0])] = times[k]
    return times

def determine_permute_matrix(set_indice_length):
    """
    permuted_label: 
                    array([ [0., 2., 1., 3., 4., 5.],
                            [1., 3., 0., 2., 5., 4.],
                            [2., 0., 4., 5., 1., 3.],
                            [3., 1., 5., 4., 0., 2.],
                            [4., 5., 2., 0., 3., 1.],
                            [5., 4., 3., 1., 2., 0.]])
    """
    length = int(set_indice_length * (set_indice_length - 1) / 2)
    num_classes = math.factorial(length)
    permuted_label = np.zeros((num_classes, math.factorial(set_indice_length)))

    permute_idx = np.array(list(permutations(range(0, set_indice_length))), dtype=np.int)
    set_indice = np.array(list(range(1, set_indice_length + 1)), dtype=np.int)

    for y in range(num_classes):
        times = time_dict(set_indice, y)
        for i in range(math.factorial(set_indice_length)):
            pemu_idx = permute_idx[i]
            pemu_set_indice = set_indice[pemu_idx]
            pemu_y = determine_triad_class_(pemu_set_indice, times)
            permuted_label[y][i] = pemu_y
        
    inverse_permute_matrix = np.zeros_like(permuted_label, dtype=np.int)
    for i in range(length):
        inverse_permute_matrix[:, i] = np.argsort(permuted_label[:, i])

    return permute_idx, permuted_label, inverse_permute_matrix
    

def order_to_label(label, l):
    for i, p in enumerate(permutations(range(1, l+1))):
        if list(p) == label:
            return i

def label_to_order_inc_one(label, l):
    for i, p in enumerate(permutations(range(1, l+1))):
        if i == label:
            return p

def permute_optimize(model, prediction, target_model, batch, optimizer, set_indice_length,
                     prevent_repeat=False):
    """ for permutation loss training。针对一个batch。
    """
    device = model_device(model)
    mse = torch.nn.MSELoss()

    # permuted batch
    permute_idx, permuted_label, inverse_permute_matrix = determine_permute_matrix(set_indice_length)
    num_graphs = batch.num_graphs
    pemu_rand = np.random.randint(0, int(math.factorial(set_indice_length)), size=num_graphs)
    batch_copy = copy.deepcopy(batch)
    permuted_batch = permute_batch(batch_copy, permute_idx, pemu_rand) # permuted batch

    # build target values
    target_model = target_model.to(device)
    l = int((set_indice_length * (set_indice_length - 1)) / 2)

    with torch.no_grad():
        encoding = target_model(permuted_batch).reshape(1, len(permuted_batch), -1)
        decoder_input = torch.zeros(len(permuted_batch), dtype=torch.float32).reshape((-1, 1, 1))
        outputs = []
        for di in range(l):
            decoder_output, decoder_hidden = model.decoder(decoder_input, encoding)
            if prevent_repeat:
                if len(outputs) > 0:
                    current_outputs = torch.cat(outputs, dim=2)
                    current_outputs = torch.cat([torch.zeros(len(batch), dtype=torch.int64).reshape((-1, 1, 1)),
                                                 current_outputs],
                                                dim=2)
                else:
                    current_outputs = torch.zeros(len(batch), dtype=torch.int64).reshape((-1, 1, 1))
                decoder_output.scatter_(2, current_outputs, -float("inf"))
            topv, topi = decoder_output.topk(1, dim=2)
            decoder_input = topi.detach().to(torch.float32)
            outputs.append(topi)

        outputs = torch.cat(outputs, dim=2).squeeze()
        permuted_prediction = []
        for i in range(len(batch)):
            permuted_prediction.append(order_to_label(outputs[i].detach().numpy().tolist(), l))
        permuted_prediction = torch.tensor(permuted_prediction, dtype=torch.int64)
        permuted_prediction = torch.nn.functional.one_hot(permuted_prediction,
                                                          math.factorial(l))

    inverse_permute_matrix = torch.LongTensor(inverse_permute_matrix).to(device)
    pemu_rand = torch.LongTensor(pemu_rand).to(device)
    for i in range(num_graphs): # align
        permuted_prediction[i] = permuted_prediction[i][inverse_permute_matrix[:, pemu_rand[i]]] # aligned with prediction
    permuted_prediction = permuted_prediction.detach() # pure tensor, no grad required

    permuted_prediction = torch.argmax(permuted_prediction, dim=1)
    new_permuted_prediction = []
    for i in range(len(batch)):
        new_permuted_prediction.append(label_to_order_inc_one(permuted_prediction[i], l))
    permuted_prediction = torch.tensor(new_permuted_prediction, dtype=torch.float32)

    # compute loss
    loss = mse(prediction, permuted_prediction)
    return loss


def optimize_model(model, target_model, dataloader, optimizer, logger, summary_writer, epoch, **kwargs):
    """ training for one epoch """
    model.train()
    device = model_device(model)
    passed_batches = 0
    update_batch = 16
    count = 0
    optimizer.zero_grad()
    train_loss = []
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='train'):
        try:
            batch = batch.to(device)
            batch_copy = copy.deepcopy(batch)
            model = model.to(device)
            label_seq = batch.y_seq
            encoding = model(batch).reshape(1, len(batch), -1)
            l = int((kwargs["set_indice_length"] * (kwargs["set_indice_length"] - 1)) / 2)

            decoder_input = torch.zeros(len(batch), dtype=torch.float32).reshape((-1, 1, 1))

            loss = 0
            outputs = []
            for di in range(l):
                decoder_output, decoder_hidden = model.decoder(decoder_input, encoding)
                loss += criterion(decoder_output.squeeze(), label_seq[:, di])
                decoder_input = label_seq[:, di].reshape((-1, 1, 1)).to(torch.float32)

                # Get prediction
                if kwargs['perm_loss'] > 0:
                    if kwargs["prevent_repeat"]:
                        if len(outputs) > 0:
                            current_outputs = torch.cat(outputs, dim=2)
                            current_outputs = torch.cat([torch.zeros(len(batch), dtype=torch.int64).reshape((-1, 1, 1)),
                                                         current_outputs],
                                                        dim=2)
                        else:
                            current_outputs = torch.zeros(len(batch), dtype=torch.int64).reshape((-1, 1, 1))
                        decoder_output.scatter_(2, current_outputs, -float("inf"))
                    topv, topi = decoder_output.topk(1, dim=2)
                    outputs.append(topi)

            if kwargs['perm_loss'] > 0:
                prediction = torch.cat(outputs, dim=2).squeeze()

            if epoch >= 10:
                weight = kwargs['perm_loss']
                if weight > 0:
                    p_loss = permute_optimize(model, prediction, target_model, batch_copy, optimizer,
                                              kwargs["set_indice_length"], kwargs["prevent_repeat"])
                    loss += p_loss * weight

            loss.backward()
            train_loss.append(loss.item() * len(batch))

            if count >= update_batch:
                optimizer.step()
                optimizer.zero_grad()
                count = 0
            else:
                count += len(batch)
            
        except Exception as e: 
            if 'CUDA out of memory' in e.args[0]:
                logger.info(f'CUDA out of memory for batch {i}, skipped.')
                passed_batches += 1
            else: 
                raise
    # update target model
    target_model.load_state_dict(model.state_dict())
    logger.info('Passed batches: {}/{}'.format(passed_batches, len(dataloader)))
    print("Train loss ", np.mean(train_loss))


class Recorder(object):
    def __init__(self, minmax: Dict, checkpoint_dir: str, time_str: str):
        """ recordes in checkpoint_dir/time_str/record.csv
        """
        self.minmax = minmax
        self.full_metrics = OrderedDict({'train': [], 'val': [], 'test': []} )
        self.time_str = time_str

        if not isinstance(checkpoint_dir, Path):
            self.checkpoint_dir = Path(checkpoint_dir)
        else: self.checkpoint_dir = checkpoint_dir        
        
        if not (self.checkpoint_dir/self.time_str).exists():
            os.makedirs((self.checkpoint_dir/self.time_str))

    def append_full_metrics(self, metrics_results, name):
        assert name in ['train', 'val', 'test']
        self.full_metrics[name].append(metrics_results)

    def save(self):
        full_metrics = copy.deepcopy(self.full_metrics)

        filename = self.checkpoint_dir/self.time_str/'record.csv'
        for key in list(full_metrics.keys()):
            full_metrics[key] = pd.DataFrame(full_metrics[key])

        for key in list(full_metrics.keys()):
            full_metrics[key] = full_metrics[key].rename(columns=lambda x:key+'_'+x)

        df = pd.concat( list(full_metrics.values()), axis=1 )
        df.to_csv(filename, float_format='%.4f', index=True, index_label='epoch' )


    @classmethod
    def load(cls, checkpoint_dir, time_str):
        filename = Path(checkpoint_dir)/time_str/'record.csv'
        assert filename.exists, 'no such file: {}'.format(filename)
        df = pd.read_csv(filename)
        return df

    def get_best_metric(self, name):
        """
        return best value and best epoch
        name: 'train', 'val', 'test'
        """
        df = pd.DataFrame( self.full_metrics[name] )
        best_metric = {}
        best_epoch = {}
        for key in df.keys():
            data = np.array(df[key])
            best_metric[key] = np.max(data) if self.minmax[key] else np.min(data)
            best_epoch[key] = np.argmax(data) if self.minmax[key] else np.argmin(data)
        return best_metric, best_epoch
    
    def get_latest_metric(self, name):
        """
        name: train, val, test
        """
        latest_metric = self.full_metrics[name][-1]
        return latest_metric

    def append_model_state(self, state_dict, metrics):
        try:
            self.model_state.append(state_dict)
        except AttributeError:
            self.model_state = []
            self.model_state.append(state_dict)

    def save_model(self):
        try:
            best_metric, best_epoch = self.get_best_metric('test')
            for key, ep in best_epoch.items():
                torch.save( self.model_state[ep], self.checkpoint_dir/self.time_str/f'{key}_state_dict')
        except AttributeError:
            pass

    def load_model(self, checkpoint_dir, time_str):
        if not isinstance(checkpoint_dir, Path):
            checkpoint_dir = Path(checkpoint_dir)
        model_state_dict = {}
        state_dict_files = checkpoint_dir.glob('*_state_dict')
        for file in state_dict_files:
            model_state_dict[str(file).rstrip('_state_dict')] = torch.load(checkpoint_dir/time_str/file)
        return model_state_dict
