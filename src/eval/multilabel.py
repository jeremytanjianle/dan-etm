"""
Evaluation code for multi-label classification
"""
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import numpy as np
import torch

def evaluate_multilabels(ypred, labels, columns=None):
    n_labels = labels.shape[-1]
    columns = columns if columns is not None else [i for i in range(n_labels)]
    ypred = ypred.cpu().detach().numpy() if type(ypred) == torch.Tensor else ypred
    
    auto_yres_test = pd.DataFrame(np.zeros((4, n_labels)),
                                index=['Precision', 'Recall', 'F1-score', 'Accuracy'],
                                columns=columns)

    for i in range(n_labels):
        precision = precision_score(labels[:,i],
                                    ypred[:, i])
        auto_yres_test.iloc[0, i] = round(precision, 4)

        recall = recall_score(labels[:,i], 
                              ypred[:, i])
        auto_yres_test.iloc[1, i] = round(recall, 4)

        f1 = f1_score(labels[:,i], 
                      ypred[:, i])
        auto_yres_test.iloc[2, i] = round(f1, 4)
        
        acc = accuracy_score(labels[:,i], 
                             ypred[:, i])
        auto_yres_test.iloc[3, i] = round(acc, 4)
        
    return auto_yres_test