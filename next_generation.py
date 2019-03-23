import codecs

import glob

import multiprocessing

import os

import pprint

import re

import nltk

import gensim.models.word2vec as w2v

from nltk.tokenize import MWETokenizer

import sklearn.manifold

import numpy as np

import pandas as pd

from nltk.corpus import stopwords

import logging

from word_extractory import article_generator_text

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

raw_corpus = 'Neutron stars, the end-stage remnants of massive stars, are high-energy objects. ' \
           'They’re usually studied in X-rays, some of the most energetic light in the universe.' \
           ' Neutron stars also give off radio emissions, most famously as pulsars. ' \
           'But now, infrared emission around a neutron star detected with the Hubble Space Telescope has sparked curiosity, ' \
           'indicating that astronomers may want to add infrared light to their neutron star-studying toolkit. ' \
           'Infrared detectors are the night-vision goggles of astronomy. ' \
           'These instruments pick up heat, which allows astronomers to punch through dust (which is cool) and view objects that are otherwise hidden from sight. ' \
           'Infrared light can also come from “reprocessed” emission, or higher-energy light that is absorbed by dust and then re-emitted at longer wavelengths. ' \
           'In a paper published September 17 in the Astrophysical Journal, a team of researchers reports the discovery of infrared emission from an area around the pulsar RX J0806.4-4123. ' \
           '“This particular neutron star belongs to a group of seven nearby X-ray pulsars — nicknamed ‘the Magnificent Seven’,” said Bettina Posselt of Penn State and the lead author of the paper in a press release. ' \
           'The Magnificent Seven, she said, “are hotter than they ought to be.” Thats already unusual, but in addition, “we observed an extended area of infrared emissions around this neutron star … the total size of which translates into about 200 astronomical units (or 2.5 times the orbit of Pluto around the Sun) at the assumed distance of the pulsar,” she said. While extended emission around neutron stars has been seen before, RX J0806.4-4123 is the first neutron star to show this type of emission only in the infrared, rather than at other wavelengths. That’s unique, and spurred the researchers to develop two possible theories for what’s going on around the distant stellar ember. One theory is that a disk of material, originally thrown out by the supernova that formed the neutron star, settled back in around the star after the explosion. That material, likely mostly dust, could be both heating the neutron star and slowing down its rotation. Emission from the dusty disk could be what astronomers are seeing in the infrared. “If confirmed as a supernova fallback disk, this result could change our general understanding of neutron star evolution,” said Posselt. It would also provide a neat explanation for the pulsar’s unusually high temperature. The second theory is less groundbreaking for neutron star studies, but still unique. Some pulsars throw off “pulsar wind” — particles accelerated by the neutron star’s massive magnetic field. As pulsars move through space, their wind can slam into interstellar material in their path, creating a shock that’s called a pulsar wind nebula. ' \
           'Pulsar wind nebulae have been observed, but usually in X-ray light. One seen only in infrared, such as could be the case around RX J0806.4-4123, “would be very unusual and exciting,” Posselt said. What’s the next step? Better infrared observations around RX J0806.4-4123 will allow the team to differentiate between their two proposed ideas. In particular, the James Webb Space Telescope will have the capability to follow up on cases such as this to provide the detail needed to determine what might be causing the emission. More in-depth infrared observations of other neutron stars, including those in the Magnificent Seven, might also provide pieces of the puzzle. “Most pulsars do not have deep [near-infrared] observations,” the team’s paper concludes, because they have long been expected to appear “unexciting” in this portion of the spectrum. “RX J0806.4–4123 is a good example that neutron stars keep surprising us.”'

raw_text2 = 'Neutron stars are remnants of stellar death so dense that they pack more than the mass of the Sun in a sphere the size of a small city. They are composed of nuclear matter produced by some types of supernovae, which occur when massive stars run out of fuel to power nuclear fusion reactions in their core and hence lose all their support against gravitational collapse. The pressure of the collapse is so great that it can be balanced only when the matter in the star is compressed to the point where neutrons and protons in atomic nuclei start pushing against each other. This is known as the neutron degeneracy pressure. Neutron stars are observed in a variety of systems: as isolated objects emitting pulses of light towards us (pulsars) and giant flares  or as binary systems with other stars, white dwarfs or even other neutron stars. All of these systems produce copious hard X-ray emission which tells us details about the masses, radii, magnetic fields and their interaction with their companions.Neutron stars have extremely strong magnetic fields. Some of them, known as magnetars have the strongest magnetic fields in the entire universe, a hundred million times stronger than the strongest man-made magnetic fields. These magnetic poles of these stars emit cones of light in radio, optical, X-ray or gamma-ray wavelengths. Much like a lighthouse, the rotation of the neutron stars periodically sweeps these cones of light in the direction of the Earth, causing us to see a pulsating star, or a pulsar.'

query = 'white dwarf'

query_to_corpus = {}

stop_words_additional = {'should', 'could', 'again', 'likewise', 'of course', 'like', 'as', 'too', 'would', 'around', 'provide',
                         'whatever', 'even', 'this', 'way'}

topic2vec = None

'''
processing data
'''

def preprocessing(raw_corpus):

    nltk.download('punkt')

    nltk.download('stopwords')

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    raw_sentences = tokenizer.tokenize(raw_corpus)

    sentences = []

    stop_words = set(stopwords.words('english'))

    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0 and raw_sentence not in stop_words:
            sentences.append(sentence_to_wordlist(raw_sentence))

    return sentences


def sentence_to_wordlist(raw):

    stop_words = set(stopwords.words('english'))

    clean = re.sub("[^a-zA-Z]"," ", raw)

    words = clean.split()

    filtered_words = list(filter(lambda x: x not in stop_words not in stop_words_additional, words))

    return filtered_words



def train_model(sentences):

    num_of_features = 200

    min_word_count = 3

    num_processors = multiprocessing.cpu_count()

    context_size = 7

    downsampling = 1e-3

    seed = 1

    topic2vec = w2v.Word2Vec(
        sg=1,
        seed=seed,
        workers=num_processors,
        size=num_of_features,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling
    )

    topic2vec.build_vocab(sentences)

    return topic2vec


def save_model(topic2vec):
    if not os.path.exists("trained"):
        os.makedirs("trained")
        #topic2vec.save(os.path.join("trained", "topic2vec.w2v"))


def process_query():

    user_query = sentence_to_wordlist(query)

    topic2vec = get_model()

    next_generation = get_next_similarity(user_query, topic2vec)

    return next_generation


def get_model():

    global topic2vec

    sentences = preprocessing(raw_corpus)

    topic2vec = train_model(sentences)

    #save_model(topic2vec)

    #return topic2vec


def get_next_similarity(query):

    global topic2vec

    filtered_queries = []

    for word in query:
        try:
            filtered_queries.append(topic2vec.most_similar(word))
        except KeyError as e:
            process_invalid_query(query)
            filtered_queries.append(topic2vec.most_similar(word))

    word_to_rating = {}

    for query in filtered_queries[0]:
        if query[0] in word_to_rating:
            word_to_rating[query[0]] = word_to_rating[query[0]] * 2
        else:
            word_to_rating[query[0]] = query[1]

    ranked_words = sorted(word_to_rating, key=word_to_rating.get, reverse=True)

    for word in ranked_words:
        if len(word) < 3 or word.__contains__('ing'):
            ranked_words.remove(word)

    return ranked_words


def process_invalid_query(word):

    global raw_corpus

    new_text_base = article_generator_text(word, 10)

    raw_corpus = str_join(raw_corpus, ' ', new_text_base)

    get_model()


def str_join(*args):
    return ''.join(map(str, args))

get_model()

print(get_next_similarity(query.split(' ')))


