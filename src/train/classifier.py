import torch
from torch import nn, optim
from torch.nn import functional as F
from ..models.dan_etm import DAN_ETM

def train_clf_one_epoch(wrapper, bow_train, labels, epoch, batchsize = 1000, clip=0, log_interval=2, verbose=1):
    """
    Train model over the data for one epoch

    args:
    ----
        bow_train: (torch.Tensor) bag of words to train over, unnormalized
        epoch: (int) for monitoring purporses
        batchsize: (int, default=1000)
        log_interval: (int, default=2) prints trinaing progress over log_interval batches
        verbose: (bool / int) to print results or not to print? 
    """
    wrapper.model.train()

    indices = torch.randperm(bow_train.shape[0])
    indices = torch.split(indices, batchsize)

    total_loss = 0

    for idx, ind in enumerate(indices):
        wrapper.optimizer_clf.zero_grad()
        wrapper.model.zero_grad()
        
        data_batch = bow_train[ind]
        labels_batch = labels[ind]
        # print(data_batch)

        bce_loss  = wrapper.model.train_clf(data_batch, labels_batch)
        #print(bce_loss)

        bce_loss.backward()
        wrapper.optimizer_clf.step()

        total_loss += bce_loss
        total_loss /= bow_train.shape[0]
    
    if verbose:
        print('*'*20)
        print('Epoch----->{} .. LR: {} .. BCE: {}'.format(
                epoch, wrapper.optimizer_clf.param_groups[0]['lr'], total_loss ))
        print('*'*20)

def train_classifier(wrapper, train_data, labels, layer_num=2, freeze_enc=True, epochs=1000, batchsize=1000, verbose=1):
    """
    
    args:
    -----
        train_data: (list) contains corpus
        labels: (np.array) 
    """
    # initialize new weights
    wrapper.model.init_clf(layer_num=layer_num, 
                        n_labels=labels.shape[-1], 
                        freeze_enc=freeze_enc)
                        
    # reinit training data and optimizer
    wrapper.bow_train = wrapper.preprocess(train_data)
    wrapper.optimizer_clf = optim.Adam(wrapper.model.downstream_clf.parameters(), 
                                    lr=wrapper.lr, 
                                    weight_decay=wrapper.wdecay)


    # the actual training
    for i in range(epochs):
        wrapper.train_clf_one_epoch(wrapper, wrapper.bow_train, labels,
                                    epoch=i, batchsize=batchsize,  
                                    verbose=verbose)

    return wrapper