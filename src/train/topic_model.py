import torch
from torch import nn, optim
from torch.nn import functional as F
from ..models.dan_etm import DAN_ETM

def train_one_epoch(wrapper, bow_train, epoch, batchsize = 1000, clip=0, log_interval=2, verbose=1):
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

    acc_loss = 0
    acc_kl_theta_loss = 0
    cnt = 0

    indices = torch.randperm(bow_train.shape[0])
    indices = torch.split(indices, batchsize)

    for idx, ind in enumerate(indices):
        wrapper.optimizer.zero_grad()
        wrapper.model.zero_grad()
        
        data_batch = bow_train[ind]
        
        recon_loss, kld_theta = wrapper.model(data_batch)
        total_loss = recon_loss + kld_theta
        total_loss.backward()
        
        # clip
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(wrapper.model.parameters(), clip)
        wrapper.optimizer.step()

        acc_loss += torch.sum(recon_loss).item()
        acc_kl_theta_loss += torch.sum(kld_theta).item()
        cnt += 1

        if idx % log_interval == 0 and idx > 0:
            cur_loss = round(acc_loss / cnt, 2) 
            cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
            cur_real_loss = round(cur_loss + cur_kl_theta, 2)
            
            if verbose:
                print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                    epoch, idx, len(indices), wrapper.optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))

    cur_loss = round(acc_loss / cnt, 2) 
    cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
    cur_real_loss = round(cur_loss + cur_kl_theta, 2)
    
    if verbose:
        print('*'*100)
        print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                epoch, wrapper.optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
        print('*'*100)


def fit_topic_model(wrapper, train_data, 
                    magnitude=None, doc_encoder_dim=800, doc_encoder_act=nn.ReLU(), enc_drop=0, clip=0,
                    seed_topic_list=None, freeze_topic_prior=False,
                    epochs = 1000, batchsize=1000, 
                    log_interval=2, verbose=1):
    """
    Train model and preprocessor

    args:
    ----
        train_data: (iterable of str) training data
        epochs: (int) rounds to train over the training data
        batchsize: (int) batch size of the training set
        verbose: (int / bool) to display training logs

    Example seed:
    ------------
    # https://github.com/vi3k6i5/GuidedLDA
    seed_topic_list = [['game', 'team', 'win', 'player', 'season', 'second', 'victory'],
                        ['percent', 'company', 'market', 'price', 'sell', 'business', 'stock', 'share'],
                        ['music', 'write', 'art', 'book', 'world', 'film'],
                        ['political', 'government', 'leader', 'official', 'state', 'country', 'american',
                        'case', 'law', 'police', 'charge', 'officer', 'kill', 'arrest', 'lawyer']]
    """
    # logic to prevent re-initialization everything again
    if wrapper.is_fitted==False:
        wrapper.bow_train = wrapper.fit_preprocess(train_data, verbose=verbose)

        if magnitude is not None:
            embeddings = wrapper.get_pretrained_word_embed(wrapper.vocab, magnitude)
            word_embed_dim = embeddings.size()[1]
        else:
            embeddings=None
            word_embed_dim=300

        wrapper.model = DAN_ETM(wrapper.num_topics, 
                        len(wrapper.vocab),
                        doc_encoder_dim = doc_encoder_dim, 
                        doc_encoder_act = doc_encoder_act,
                        word_embed_dim = word_embed_dim,  
                        embeddings = embeddings, 
                        enc_drop=enc_drop).to(wrapper.device)
        wrapper.optimizer = optim.Adam(wrapper.model.parameters(), 
                                    lr=wrapper.lr, 
                                    weight_decay=wrapper.wdecay)
        wrapper.is_fitted = True
    else:
        wrapper.bow_train = wrapper.preprocess(train_data)

    """
    set initial weights for topic representations
    """
    if (seed_topic_list is not None) & (magnitude is not None):

        magnitude = wrapper.get_magnitude(magnitude)

        for idx, seed_topic in enumerate(seed_topic_list):
            # get mean of seeded word embeddings
            init_topic = [torch.from_numpy(magnitude.query(seed)).type(torch.cuda.FloatTensor).view(1,-1) 
                            for seed in seed_topic]
            init_topic = torch.cat(init_topic, axis=0).mean(axis=0)
            
            # change weights of the topic embedding
            # https://discuss.pytorch.org/t/how-to-manually-set-the-weights-in-a-two-layer-linear-model/45902
            with torch.no_grad():
                wrapper.model.topic_embed.weight[idx,:] = init_topic
        
        if freeze_topic_prior: wrapper.model.topic_embed.weight.requires_grad = False

    # the actual training
    for i in range(epochs):
        train_one_epoch(wrapper, wrapper.bow_train, 
                        epoch=i, batchsize=batchsize,  
                        log_interval = log_interval, verbose=verbose)

    return wrapper