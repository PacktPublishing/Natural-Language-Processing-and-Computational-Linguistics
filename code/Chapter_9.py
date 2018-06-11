# pre-processing tips

# adding stop words in spacy
# remember to load your appropriate language model

my_stop_words = [u'say', u'\'s', u'Mr', u'be', u'said', u'says', u'saying']
for stopword in my_stop_words:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True

# adding logging

import logging
logging.basicConfig(filename='logfile.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# document - topic proportions
ldamodel[corpus[0]] 

# printing first topic

ldamodel.show_topics()[1]

texts = [['bank','river','shore','water'],
        ['river','water','flow','fast','tree'],
        ['bank','water','fall','flow'],
        ['bank','bank','water','rain','river'],
        ['river','water','mud','tree'],
        ['money','transaction','bank','finance'],
        ['bank','borrow','money'], 
        ['bank','finance'],
        ['finance','money','sell','bank'],
        ['borrow','sell'],
        ['bank','loan','sell']]

model.get_term_topics('water')
model.get_term_topics('finance')

bow_water = ['bank','water','bank']
bow_finance = ['bank','finance','bank']
bow = model.id2word.doc2bow(bow_water) # convert to bag of words format first
doc_topics, word_topics, phi_values = model.get_document_topics(bow, per_word_topics=True)

# coherence models

lsi_coherence = CoherenceModel(topics=lsitopics[:10], texts=texts, dictionary=dictionary, window_size=10)
hdp_coherence = CoherenceModel(topics=hdptopics[:10], texts=texts, dictionary=dictionary, window_size=10)
lda_coherence = CoherenceModel(topics=ldatopics, texts=texts, dictionary=dictionary, window_size=10)

# train two models, one poorly trained (1 pass), and one trained with more passes (50 passes)

print(goodcm.get_coherence())
print(badcm.get_coherence())


c_v = []
for num_topics in range(1, limit):
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary,          coherence='c_v')
        c_v.append(cm.get_coherence())

# visualisation 

import pyLDAvis.gensim
pyLDAvis.gensim.prepare(lda, corpus, dictionary)

