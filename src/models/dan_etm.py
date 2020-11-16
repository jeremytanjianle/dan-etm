"""
Implementing:

1. Seeded topic representation
2. DAN with word dropout
    https://discuss.pytorch.org/t/implementing-word-dropout/23700
"""

import torch
import torch.nn.functional as F 
from torch import nn
import numpy as np 
import math 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Highway(nn.Module):
    """
    Highway networks as implemented in:
    https://github.com/hengruo/QANet-pytorch/blob/master/models.py
    """
    def __init__(self, layer_num: int, size: int):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])

    def forward(self, x):
        # x = x.transpose(1, 2)
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        # x = x.transpose(1, 2)
        return x



class DAN_ETM(nn.Module):
    def __init__(self, num_topics, vocab_size, 
                doc_encoder_act=nn.ReLU(),
                word_embed_dim = 300,  
                embeddings=None, 
                enc_drop=0.5,
                doc_encoder_dim=None):
        super(DAN_ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.word_embed_dim = word_embed_dim
        self.enc_drop = enc_drop
        self.t_drop = nn.Dropout(enc_drop)
        self.downstream_clf_init = False

        self.doc_encoder_act = doc_encoder_act
        
        """
        define the word embedding matrix rho
        rho to be of shape (Vocab x Word embedding size)
        """
        if embeddings is None:
            self.word_embed = nn.Linear(word_embed_dim, vocab_size, bias=False).weight
        else:
            self.word_embed = embeddings.clone().float().requires_grad_(True).to(device)

        ## define the matrix containing the topic embeddings
        # self.topic_embed = nn.Parameter(torch.randn(word_embed_dim, num_topics))
        self.topic_embed = nn.Linear(word_embed_dim, num_topics, bias=False)
    
        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
                Highway(layer_num=2, size=word_embed_dim),
                nn.Linear(word_embed_dim, word_embed_dim), 
            )
        self.mu_q_theta = nn.Linear(word_embed_dim, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(word_embed_dim, num_topics, bias=True)

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.
        input: bows
                batch of bag-of-words...tensor of shape (batch size x Vocab)
        output: mu_theta, log_sigma_theta
        """
        # normalize bag of words
        bows = bows/bows.sum(1).unsqueeze(1)

        averaged_word_embeddings = torch.mm(bows , self.word_embed)

        # encode bow into document encoding
        q_theta = self.q_theta(averaged_word_embeddings)
        if (self.enc_drop > 0) & self.downstream_clf_init is False:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kld_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kld_theta

    def get_word_topic_matrix(self):
        """
        Retrieves beta which "denotes a traditional topic; ie, a distribution over all the words." 
        Beta would be shape (vocab_size, num_topics)

        There is an elegant math where the word distrbution over the topic is derived
        by the softmax of the dot product of the word embedding and the topic embedding.
        i.e. the more the topic and the word agree with each other, the more likely the topic 
        generates the words.

        Shapes:
            topic_embed.shape = word_embed_dim, num_topics
            word_embed.weight.shape = vocab_size, word_embed_dim
            softmax( topic_embed(word_embed) ).shape = vocab_size, num_topics
        """
        topic_word_agreement = self.topic_embed(self.word_embed)
        word_distribution_over_topics = F.softmax(topic_word_agreement, dim=0).transpose(1, 0) ## softmax over vocab dimension
        return word_distribution_over_topics

    def get_topic_proportions(self, bows):
        """
        Derive topic proportion from bag of words.

        Topic proportion matrix denoted as Theta in the paper.
        Authors: "Theta denotes the (transformed) topic proportions", that is, 
        it allocates to each document the proportion of it that belongs to which topic

        For inference, the transformed topic proportions are just the softmax-ed mu_theta

        theta.shape = (documents or len(normalized_bows), number of topics)
        """
        # get logits and topic proportions
        mu_theta, logsigma_theta, kld_theta = self.encode(bows)
        z = self.reparameterize(mu_theta, logsigma_theta) if not self.downstream_clf_init else mu_theta
        topic_proportions = F.softmax(z, dim=-1) 
        return topic_proportions, kld_theta

    def decode(self, topic_proportions, word_topic_matrix):
        """
        Reconstruct a document based on topic proportions and topic-word distributions
        """
        res = torch.mm(topic_proportions, word_topic_matrix)
        preds = torch.log(res+1e-6)
        return preds 

    def forward(self, bows, topic_proportions=None, aggregate=True):
        ## get \theta
        if topic_proportions is None:
            topic_proportions, kld_theta = self.get_topic_proportions(bows)
        else:
            kld_theta = None

        ## get word_topic_matrix \beta
        word_topic_matrix = self.get_word_topic_matrix()

        ## get prediction loss
        preds = self.decode(topic_proportions, word_topic_matrix)
        recon_loss = -(preds * bows).sum(1)
        if aggregate:
            recon_loss = recon_loss.mean()
        return recon_loss, kld_theta


    """
    INIT DOWNSTREAM CLASSIFIER
    """
    def init_clf(self, layer_num=2, n_labels=5, freeze_enc = True):
        """
        Initialize downstream classifier and freeze all encoder weights
        """
        if freeze_enc:
            self.word_embed.requires_grad = False
            self.topic_embed.requires_grad = False
            self.q_theta.requires_grad = False
            self.mu_q_theta.requires_grad = False
            self.logsigma_q_theta.requires_grad = False
        self.downstream_clf_init = True

        # init classifier
        self.downstream_clf = nn.Sequential(
                Highway(layer_num=layer_num, size=self.num_topics),
                nn.Linear(self.num_topics, n_labels), 
            ).to(device)
        return self

    def classify(self, bow):
        """
        Classify document according to labels
        It really gives the probability
        """
        mu, sigma, kld = self.encode(bow)
        logits = self.downstream_clf(mu)
        yhat = torch.sigmoid(logits)
        return yhat

    def train_clf(self, bow, labels):
        """

        """
        mu, sigma, kld = self.encode(bow)
        logits = self.downstream_clf(mu)
        labels = torch.from_numpy(labels.astype(np.int64)) # input is np.array, convert to torch.Tensor
        loss = nn.BCEWithLogitsLoss()(logits, labels.type_as(logits)) # ensure labels not longtensor
        return loss
