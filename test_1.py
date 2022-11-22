from flask import Flask
from flask import jsonify
from flask import request
app = Flask(__name__)

import heapq 
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
import tensorflow as tf
import pickle
from model import Model
from nltk.tokenize import word_tokenize
import re
import collections
import pickle
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec


#train_article_path = "sumdata/train/train.article.txt"
#train_title_path = "sumdata/train/train.title.txt"
valid_article_path = "sumdata/train/valid.article.filter.txt"
valid_title_path = "sumdata/train/valid.title.filter.txt"

def tag(sentence):
  words = word_tokenize(sentence)
  words = pos_tag(words)
  return words

def paraphraseable(tag):
  return tag.startswith('VB')

def pos(tag):
  if tag.startswith('NN'):
    return wn.NOUN
  elif tag.startswith('V'):
    return wn.VERB

def synonyms(word, tag):
    lemma_lists = [ss.lemmas() for ss in wn.synsets(word, pos(tag))]
    lemmas = [lemma.name() for lemma in sum(lemma_lists, [])]
    return set(lemmas)

def synonymIfExists(sentence):
  for (word, t) in tag(sentence):
    if paraphraseable(t):
      syns = synonyms(word, t)
      if syns:
        if len(syns) > 1:
          yield list(syns)[0]
          continue
    yield word

def paraphrase(sentence):
  return [x for x in synonymIfExists(sentence)]

def the_summary(sentence):
    x = ' '.join(paraphrase(sentence))
    return x 

def get_imp_lines(article_text):
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)  
    article_text = re.sub(r'\s+', ' ', article_text)  

    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )  
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)  

    sentence_list = nltk.sent_tokenize(article_text)

    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}  
    for word in nltk.word_tokenize(formatted_article_text):  
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1


    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    summary_sentences = heapq.nlargest(2, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    return summary

def clean_str(sentence):
    sentence = re.sub("[#.]+", "#", sentence)
    return sentence


def get_text_list(data):
    return [clean_str(x.strip()) for x in data]

def build_dict(step, toy=False):
    if step == "valid":
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))

    article_max_len = 50
    summary_max_len = 15

    return word_dict, reversed_dict, article_max_len, summary_max_len


def build_dataset(step, word_dict, article_max_len, summary_max_len, data,toy=False):
    if step == "valid":
        article_list = get_text_list(data)
    else:
        raise NotImplementedError
    print("==============================================================================================")
    print(article_list[0])
    print(len(article_list))
    x = [word_tokenize(d) for d in article_list]
    x = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in x]
    x = [d[:article_max_len] for d in x]
    x = [d + (article_max_len - len(d)) * [word_dict["<padding>"]] for d in x]
    
    if step == "valid":
        return x
    else:        
        y = [word_tokenize(d) for d in title_list]
        y = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in y]
        y = [d[:(summary_max_len - 1)] for d in y]
        return x, y


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]

with open("args.pickle", "rb") as f:
    args = pickle.load(f)
print("Loading dictionary...")
word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("valid", args.toy)
print("Loading saved model...")
model = Model(reversed_dict, article_max_len, summary_max_len, args, forward_only=True)
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state("./saved_model/")
# with tf.Session() as sess:
#     saver.restore(sess, ckpt.model_checkpoint_path)

@app.route('/hello',methods=['POST'])
def hello():
    message = request.get_json(force=True)
    name = message['name']
    # response = {
    #   'greeting':'hello, '+name+'!'
    # }
    # return jsonify(response)

    pass_article_list = [name]

    # with open("args.pickle", "rb") as f:
    #     args = pickle.load(f)
    #pass_article_list = ["us business leaders lashed out wednesday at legislation that would penalize companies for employing illegal immigrants ."]
    # print("Loading dictionary...")
    # word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("valid", args.toy)
    print("Loading validation dataset...")
    valid_x = build_dataset("valid", word_dict, article_max_len, summary_max_len, pass_article_list, args.toy)
    valid_x_len = [len([y for y in x if y != 0]) for x in valid_x]

    with tf.Session() as sess:
        # print("Loading saved model...")
        # model = Model(reversed_dict, article_max_len, summary_max_len, args, forward_only=True)
        # saver = tf.train.Saver(tf.global_variables())
        # ckpt = tf.train.get_checkpoint_state("./saved_model/")
        saver.restore(sess, ckpt.model_checkpoint_path)

        batches = batch_iter(valid_x, [0] * len(valid_x), args.batch_size, 1)

        print("Generating summaries")
        for batch_x, _ in batches:
            batch_x_len = [len([y for y in x if y != 0]) for x in batch_x]

            valid_feed_dict = {
                model.batch_size: len(batch_x),
                model.X: batch_x,
                model.X_len: batch_x_len,
            }

            prediction = sess.run(model.prediction, feed_dict=valid_feed_dict)
            prediction_output = [[reversed_dict[y] for y in x] for x in prediction[:, 0, :]]

            #with open("result.txt", "a") as f:
            for line in prediction_output:
                summary = list()
                for word in line:
                    if word == "</s>":
                        break
                    if word not in summary:
                        summary.append(word)
                summary = the_summary(get_imp_lines(name)) if len(name) > 300 else " ".join(summary)
                print(summary)

        #print(summary)

    response = {
        'article': 'Your Article:'+ name ,
        'greeting':" Summary : "+summary
    }
    return jsonify(response)