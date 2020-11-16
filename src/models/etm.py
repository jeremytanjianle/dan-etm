import torch
import torch.nn.functional as F 
import numpy as np 
import math 

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ETM(nn.Module):
    def __init__(self, num_topics, vocab_size, 
                doc_encoder_dim = 800, 
                doc_encoder_act=nn.ReLU(),
                word_embed_dim = 300,  
                embeddings=None, 
                enc_drop=0.5):
        super(ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.doc_encoder_dim = doc_encoder_dim
        self.word_embed_dim = word_embed_dim
        self.enc_drop = enc_drop
        self.t_drop = nn.Dropout(enc_drop)

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
        self.topic_embed = nn.Linear(word_embed_dim, num_topics, bias=False)#nn.Parameter(torch.randn(word_embed_dim, num_topics))
    
        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
                nn.Linear(vocab_size, doc_encoder_dim), 
                self.doc_encoder_act,
                nn.Linear(doc_encoder_dim, doc_encoder_dim),
                self.doc_encoder_act,
            )
        self.mu_q_theta = nn.Linear(doc_encoder_dim, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(doc_encoder_dim, num_topics, bias=True)

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

        # encode bow into document encoding
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
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
        z = self.reparameterize(mu_theta, logsigma_theta)
        topic_proportions = F.softmax(z, dim=-1) 
        return topic_proportions, kld_theta

    def decode(self, topic_proportions, word_topic_matrix):
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