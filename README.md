# DAN - ETM Topic Model

DAN-ETM is an interpretable neural topic model that aims to be versatile in classification, auto-labelling and interpretability.

Features of the generated topics include:
1. __Interpretable:__ DAN-ETM takes a word embedding and a corpus and derives a topic representation in the same space as the word embedding. In english, you can literally see how 'close' a topic is to a word by the cosine distance between their embeddings. This is quite different from co-occurance based topic models such as LDA.
2. __Semantically tight__: Similarly, because topics are built on their semantic similarity to various words, the relation between hot topic words are strong. No more will you see the words 'garden' and 'insurance' be related to each other simply because they co-occur. 



## Datasets

There are two datasets mainly used within this repo:
1. 20 News Group dataset as retrievable from Sklearn
2. IMDB movies dataset
3. Spam data on YouTube


## Sample usage

DAN-ETM is an end-to-end model with various methods to interpret and study, save and load, and classify and label.

__Training__
```
# fetch corpus
>>> from sklearn.datasets import fetch_20newsgroups
>>> train_data = fetch_20newsgroups(subset='train').data # from sklearn
>>> test_data = fetch_20newsgroups(subset='test').data

# Fit model
>>> from src.wrapper import end2end
>>> etm = end2end(min_df=0.005, num_topics=50)
>>> etm.fit(train_data, 
            magnitude = 'GoogleNews-vectors-negative300.magnitude',
            verbose=1)
>>> print( etm.evaluate() )
>>> etm.save_model("dan_etm_20ng_500topics_mindf0.005")
```

__Interpreting__  
```
>>> etm.top_k_words()
Topic 0: ['distribution', 'netcom', 'access', 'posting', 'nntp', 'net', 'freenet', 'digex', 'services']
Topic 1: ['years', 'time', 'ago', 'year', 'long', 'back', 'good', 'times', 'days']
...
Topic 5: ['windows', 'software', 'dos', 'version', 'unix', 'ibm', 'ms', 'pub', 'comp']
...
Topic 7: ['israel', 'turkish', 'jews', 'armenian', 'israeli', 'armenians', 'jewish', 'war', 'people']
...
Topic 35: ['space', 'nasa', 'gov', 'earth', 'launch', 'moon', 'data', 'orbit', 'jpl']
...
```

A feature of the model is how semantically tight the topics are. Because it no longer fits on word co-occurance, which may sometimes be unintuitive, it may cluster words based on the word representation. 
```
>>> etm.visualize_topic_embeddings(topic_idx = 7, n=17, alpha=0.9, size=35)
```  
<img src='img/interpreting topic representation-israel topic.png'>  

## Semi-Autolabelling and Active Learning

Topic models may be used for auto-labelling of documents, using seeding. 
```
>>> from src.wrapper import end2end
>>> import pandas as pd

>>> movies = pd.read_csv('../data/movies/movies.csv')
>>> train_data = movies.overview.tolist()

>>> seed_topic_list=[['action','spy','mafia','fighter','warrior', 'police', 'cop'],
>>>                  ['romance','sex'],
>>>                  ['science_fiction'],
>>>                  ['thriller','deadly','tension'],
>>>                  ['war','military','army'],
>>>                  ['western','gunslinger', 'gunman','sheriff'],
>>>                  ['movie','starring','stars','actors','film']
>>>                 ]

>>> etm = end2end(num_topics=7, min_df=0.001)
>>> etm.fit_topic_model(movies.overview.tolist(), 
>>>                     magnitude = 'GoogleNews-vectors-negative300.magnitude',
>>>                     seed_topic_list=seed_topic_list,
>>>                     freeze_topic_prior=True,
>>>                     epochs=1000,
>>>                     verbose=1)

>>> etm.top_k_words(n=10)
```
The result is found below. 
```
Topic 0: ['cop', 'gangster', 'cops', 'fighter', 'policeman', 'thug', 'spy', 'assassin', 'mafia']
Topic 1: ['sex', 'romance', 'romantic', 'romances', 'sexual', 'sexuality', 'marriage', 'erotic', 'fling']
Topic 2: ['fiction', 'novels', 'sci', 'futuristic', 'genre', 'noir', 'supernatural', 'horror', 'movies']
Topic 3: ['thriller', 'tension', 'deadly', 'drama', 'intrigue', 'suspense', 'tense', 'tensions', 'mayhem']
Topic 4: ['military', 'army', 'war', 'troops', 'soldiers', 'naval', 'navy', 'forces', 'colonel']
Topic 5: ['gunman', 'sheriff', 'lawman', 'gunslinger', 'western', 'bandit', 'robber', 'gunfighter', 'cowboy']
Topic 6: ['movie', 'film', 'actors', 'films', 'starring', 'movies', 'actor', 'stars', 'filmmakers']
```
Autolabelling can also be done by feeding AI Singapore's Corgi into DAN-ETM.  
DAN-ETM uses Uncertainty Sampling to maximize the impact of labelling efforts.

__Classification__  
DAN-ETM can also be used to classify documents. The gist of it is to use the topic model as an __encoder__ and to train a __decoder__ on top of it. Below, are the in-sample and out-of-sample performances on the movie dataset, which is much better than LDA and NMF benchmarks.  

## References
Dieng, Adji B., Francisco JR Ruiz, and David M. Blei. "Topic modeling in embedding spaces." arXiv preprint arXiv:1907.04907 (2019).

Iyyer, Mohit, et al. "Deep unordered composition rivals syntactic methods for text classification." Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). 2015.

Srivastava, Rupesh Kumar, Klaus Greff, and Jürgen Schmidhuber. "Highway networks." arXiv preprint arXiv:1505.00387 (2015).

Dieng, Adji B., Francisco JR Ruiz, and David M. Blei. "The dynamic embedded topic model." arXiv preprint arXiv:1907.05545 (2019).

He, Junxian, et al. "Efficient correlated topic modeling with topic embedding." Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2017.