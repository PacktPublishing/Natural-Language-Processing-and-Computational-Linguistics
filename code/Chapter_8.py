# we need to first set up the text and corpus as it was done in section 3.3
# this refers to the code set-up in the Chapter 3

from gensim.models import LdaModel

ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)
ldamodel.show_topics()

lsimodel = LsiModel(corpus=corpus, num_topics=10, id2word=dictionary)
lsimodel.show_topics(num_topics=5)  # Showing only the top 5 topics

hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)
hdpmodel.show_topics()




from sklearn.decomposition import NMF, LatentDirichletAllocation
nmf = NMF(n_components=no_topic).fit(tfidf_corpus)
lda = LatentDirichletAllocation(n_topics=no_topics).fit(tf_corpus)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])

display_topics(nmf, tfidf_feature_names, no_top_words)
