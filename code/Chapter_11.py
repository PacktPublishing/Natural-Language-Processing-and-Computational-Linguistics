# make sure to have appropriate gensim installations and imports done

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
        ['bank', 'loan', 'sell']

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = TfidfModel(corpus)
model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=2)

model.show_topics()

doc_water = ['river', 'water', 'shore']
doc_finance = ['finance', 'money', 'sell']
doc_bank = ['finance', 'bank', 'tree', 'water']

bow_water = model.id2word.doc2bow(doc_water)   
bow_finance = model.id2word.doc2bow(doc_finance)   
bow_bank = model.id2word.doc2bow(doc_bank)   

lda_bow_water = model[bow_water]
lda_bow_finance = model[bow_finance]
lda_bow_bank = model[bow_bank]

tfidf_bow_water = tfidf[bow_water]
tfidf_bow_finance = tfidf[bow_finance]
tfidf_bow_bank = tfidf[bow_bank]

from gensim.matutils import kullback_leibler, jaccard, hellinger

hellinger(lda_bow_water, lda_bow_finance)
hellinger(lda_bow_finance, lda_bow_bank)
hellinger(lda_bow_bank, lda_bow_water)

hellinger(lda_bow_finance, lda_bow_water)
kullback_leibler(lda_bow_water, lda_bow_bank)
kullback_leibler(lda_bow_bank, lda_bow_water)


jaccard(bow_water, bow_bank)
jaccard(doc_water, doc_bank)
jaccard(['word'], ['word'])

def make_topics_bow(topic):
    # takes the string returned by model.show_topics()
    # split on strings to get topics and the probabilities
    topic = topic.split('+')
    # list to store topic bows
    topic_bow = []
    for word in topic:
        # split probability and word
        prob, word = word.split('*')
        # get rid of spaces
        word = word.replace(" ","")
        # convert to word_type
        word = model.id2word.doc2bow([word])[0][0]
        topic_bow.append((word, float(prob)))
    return topic_bow


topic_water, topic_finance = model.show_topics()
finance_distribution = make_topics_bow(topic_finance[1])
water_distribution = make_topics_bow(topic_water[1])

hellinger(water_distribution, finance_distribution)

from gensim import similarities

index = similarities.MatrixSimilarity(model[corpus])
sims = index[lda_bow_finance]
print(list(enumerate(sims)))

sims = sorted(enumerate(sims), key=lambda item: -item[1])

for doc_id, similarity in sims:
    print texts[doc_id], similarity

from gensim.summarization import summarize
print (summarize(text))

print (summarize(text, word_count=50))

from gensim.summarization import keywords

print (keywords(text))

from gensim.summarization import mz_keywords
mz_keywords(text,scores=True,weighted=False,threshold=1.0)

