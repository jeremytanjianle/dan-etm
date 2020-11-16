import torch 
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

def nearest_neighbors(word, embeddings, vocab):
    """
    Find nearest neighbours to words in terms of word embeddings
    """
    vectors = embeddings.data.cpu().numpy() 
    index = vocab.index(word)
    print('vectors: ', vectors.shape)
    query = vectors[index]
    print('query: ', query.shape)

    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom

    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]

    nearest_neighbors = mostSimilar[:20]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors


def visualize_topic_embeddings(wrapper, topic_idx=0, n=15, alpha=0.7, size=35):
    """
    Visualize topics and closest words in 2D scatterplot (TSNE)

    args:
    -----
        topic_idx: (int) index of topic
        n: (int) top words of that topic
        alpha: (float) alpha of the words in scatter plots 
                        Not the points but the word annotations
        size: (int) font size of annotations
    """

    #  get a list of closest words in topic
    with torch.no_grad():
        gammas = wrapper.model.get_word_topic_matrix()
        gamma = gammas[topic_idx]
        top_words = list(gamma.cpu().numpy().argsort()[-n:][::-1])
        topic_words = [wrapper.vocab[a] for a in top_words]
        print('Topic {}: {}'.format(topic_idx, topic_words))

    # word representation as numpy array
    word_embeddings = [wrapper.get_word_representation(topic_word)[0].cpu().detach().numpy()
                    for topic_word in topic_words]
    topic_embedding = wrapper.model.topic_embed.weight[24].cpu().detach().numpy()
    word_embeddings.append(topic_embedding)
    word_embeddings = np.stack(word_embeddings)

    topic_words.append(f'topic {topic_idx}')

    # compress with tsne
    tsne_wp_3d = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=12)
    embeddings_ak_2d = tsne_wp_3d.fit_transform(word_embeddings)

    # plot in matplotlib
    plt.style.use('seaborn')
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, 1))
    x = embeddings_ak_2d[:,0]
    y = embeddings_ak_2d[:,1]
    plt.scatter(x, y, c=colors, label='label')
    for i, word in enumerate(topic_words):
        c = 'red' if 'topic' in word else 'black'
        plt.annotate(word, alpha=alpha, xy=(x[i], y[i]), xytext=(5, 2), c=c,
                    textcoords='offset points', ha='right', va='bottom', size=size)
    plt.legend(loc=4)
    plt.show()