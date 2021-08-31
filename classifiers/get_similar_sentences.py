# checks cosine similarity between given sentence and the wikipedia corpus and returns top k similar sentences

from sentence_transformers import SentenceTransformer, util
import numpy as np
import nltk

# create model for generating embeddings
model = SentenceTransformer('stsb-roberta-large')

# read context data
raw_text = open('../data/wikipedia_context_files.txt', 'r', encoding='UTF-8').read()
raw_text = raw_text.replace('\n', '')
context = nltk.sent_tokenize(raw_text)

# get embeddings of context
print('Generating context embedding for sentence similarity')
context_embeddings = model.encode(context, convert_to_tensor=True)


def get_similar_sentences(sentence, num_sents=1):
    # get embeddings of sentence
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)

    # compute similarity scores of the sentence with the corpus
    cos_scores = util.pytorch_cos_sim(sentence_embedding, context_embeddings).cpu()[0]

    # Sort the results in decreasing order and get the first k_sents
    top_results = np.argpartition(-cos_scores, range(num_sents))[0:num_sents]

    return [context[i] for i in top_results]

# get_similar_sentences('barack obama met donald trump', 2)
