import torch 
import numpy as np

def get_topic_diversity(beta, topk):
    """
    calculate topic diversity from word_topic_matrix, beta
    """
    if type(beta) == torch.Tensor:
        beta = beta.cpu().detach().numpy()

    num_topics = beta.shape[0]
    list_w = np.zeros((num_topics, topk))
    for k in range(num_topics):
        idx = beta[k,:].argsort()[-topk:][::-1]
        list_w[k,:] = idx
    n_unique = len(np.unique(list_w))

    TD = n_unique / (topk * num_topics)
    #print('Topic diveristy is: {}'.format(TD))

    return TD

def normalized_pointwise_MI(bow, i, j):
    """
    Normalized pointwise MI as specified in the paper
    for one topic
    
    args:
    ----
        bow: (np.array) Must be np array and not torch.Tensor
        i: (int) word position to look up on bow
        j: (int) word position to look up on bow
    """
    incidence_i, incidence_j = (bow[:,i]==1), (bow[:,j]==1)

    P_i = incidence_i.sum() / len(bow)
    P_j = incidence_j.sum() / len(bow)
    P_ij = (incidence_i & incidence_j).sum() / len(bow)

    if P_ij==0:
        return -1
    
    normalized_pointwise_MI = np.log(P_ij/P_i/P_j ) / -np.log(P_ij )
    return normalized_pointwise_MI

def get_topic_coherence(bow, word_topic_matrix):
    """
    Calculates topic coherence as per paper suggests
    """
    # convert to numpy array for ease
    bow = bow.sign().cpu().detach().numpy()
    
    topic_coherence = 0
    for topic in word_topic_matrix:
        # get top word idx for topic
        top_10_words_idx = topic.argsort()[-10:].cpu().detach().numpy()[::-1]
        MI_one_topic = np.mean([normalized_pointwise_MI(bow, i, j) 
                                for i in top_10_words_idx 
                                for j in top_10_words_idx 
                                if i>j])
        topic_coherence += MI_one_topic
    topic_coherence /= len(word_topic_matrix)
    
    return topic_coherence

