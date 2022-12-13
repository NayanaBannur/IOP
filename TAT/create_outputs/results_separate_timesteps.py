import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from TAT.train_pred_t import acc_metric, compute_metric, f1_score_metric

metrics = {'acc': acc_metric, 'macro_f1': f1_score_metric}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

accs = []
f1s = []

for t in range(6):
    path = os.path.join(ROOT_DIR, f'results/CollegeMsg-4-t-{t}/')
    full_metrics = []
    for epoch in range(50):
        epoch_results = {}
        for split in ['val', 'test']:
            predictions = torch.load(path + f'{split}_predictions_{epoch}.pt')
            labels = torch.load(path + f'{split}_labels_{epoch}.pt')
            results = compute_metric(predictions, labels, metrics, n=4)
            if 'kendall_tau' in metrics:
                results['kendall_tau correlation'] = results['kendall_tau'].correlation
                results['kendall_tau (p-value)'] = results['kendall_tau'].pvalue
                del results['kendall_tau']
            results = {split + '_' + k: v for k, v in results.items()}
            epoch_results.update(results)
        full_metrics.append(epoch_results)
    full_metrics = pd.DataFrame(full_metrics)
    acc_idx = np.argpartition(-full_metrics["val_acc"], 1)[:1]
    test_acc = full_metrics["test_acc"][acc_idx]
    f1_idx = np.argpartition(-full_metrics["val_macro_f1"], 1)[:1]
    test_f1 = full_metrics["test_macro_f1"][f1_idx]
    accs.append(np.mean(test_acc))
    f1s.append(np.mean(test_f1))

metrics_df = pd.DataFrame(list(zip(accs, f1s)),
                          columns=['Accuracy', 'Macro F1-score'])
print(metrics_df)
metrics_df = metrics_df.stack().reset_index().rename(columns={"level_0": "Timestep", "level_1": "Metric", 0: "Value"})
metrics_df['Timestep'] += 1

x = list(range(1, 7))
plt.bar(x, accs)
plt.xlabel('Timestep')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('timesteps_acc.png', bbox_inches='tight')
plt.clf()

plt.bar(x, f1s)
plt.xlabel('Timestep')
plt.ylabel('Macro F1-score')
plt.tight_layout()
plt.savefig('timesteps_f1.png', bbox_inches='tight')
plt.clf()

g = sns.catplot(
    data=metrics_df, kind="bar",
    x="Timestep", y="Value", hue="Metric", palette='Blues', legend=False
)
g.fig.set_size_inches(8, 5)
g.set_axis_labels("Timestep", "")
plt.legend(loc='upper right')

g.figure.savefig("timesteps.png", bbox_inches='tight')
