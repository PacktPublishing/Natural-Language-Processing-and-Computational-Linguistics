# be sure to make appropriate imports and installations

from gensim.models import word2vec

sentences = word2vec.Text8Corpus('text8') 
model = word2vec.Word2Vec(sentences, size=200, hs=1)

print(model)
model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)[0]

model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])

model.wv['computer']model.save("text8_model")

model.save("text8_model")
model = word2vec.Word2Vec.load("text8_model")

model.wv.doesnt_match("breakfast cereal dinner lunch".split())

model.wv.similarity('woman', 'man')

model.wv.similarity('woman', 'cereal')
 
model.wv.distance('man', 'woman')


word_vectors = model.wv
del model

model.wv.evaluate_word_pairs(os.path.join(module_path, 'test_data','wordsim353.tsv'))
model.wv.accuracy(os.path.join(module_path, 'test_data', 'questions-words.txt'))

from gensim.models import KeyedVectors
# load the google word2vec model
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)

# LEE corpus

test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
lee_test_file = test_data_dir + os.sep + 'lee.cor'

def read_corpus(file_name, tokens_only=False):
    with smart_open.smart_open(file_name) as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

train_corpus = list(read_corpus(lee_train_file))
test_corpus = list(read_corpus(lee_test_file, tokens_only=True))

model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=100)
model.build_vocab(train_corpus) 
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)

models = [
    # PV-DBOW 
    Doc2Vec(dm=0, dbow_words=1, size=200, window=8, min_count=10, iter=50),
    
# PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, size=200, window=8, min_count=10, iter =50),
]

models[0].build_vocab(documents)
models[1].reset_from(models[0])

for model in models:
   model.train(documents, total_examples=model.corpus_count, epochs=model.iter)

from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
new_model = ConcatenatedDoc2Vec(models[0], models[1])

inferred_vector = model.infer_vector(train_corpus[0].words)
sims = model.docvecs.most_similar([inferred_vector])
print(sims)

# glove

from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

from gensim.models import KeyedVectors
filename = 'glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)

from gensim.models.fasttext import FastText

ft_model = FastText(size=100)
ft_model.build_vocab(data)
model_gensim.train(data, total_examples=ft_model.corpus_count, epochs=ft_model.iter)


from gensim.models.wrappers.fasttext import FastText

# Set FastText home to the path to the FastText executable
ft_home = '/home/bhargav/Gensim/fastText/fasttext'
# train the model
model_wrapper = FastText.train(ft_home, train_file)

print('dog' in model.wv.vocab)
print('dogs' in model.wv.vocab)

print('dog' in model)
print('dogs' in model)

from gensim.models.wrappers import Wordrank

wordrank_path = 'wordrank' # path to Wordrank directory
out_dir = 'model' # name of output directory to save data to
data = '../../gensim/test/test_data/lee.cor' # sample corpus

model = Wordrank.train(wordrank_path, data, out_dir, iter=21, dump_period=10)


varembed_vectors = '../../gensim/test/test_data/varembed_leecorpus_vectors.pkl'
model = varembed.VarEmbed.load_varembed_format(vectors=varembed_vectors)


morfessors = '../../gensim/test/test_data/varembed_leecorpus_morfessor.bin'
model = varembed.VarEmbed.load_varembed_format(vectors=varembed_vectors, morfessor_model=morfessors)

import os

poincare_directory = os.path.join(os.getcwd(), 'docs', 'notebooks', 'poincare')
data_directory = os.path.join(poincare_directory, 'data')
wordnet_mammal_file = os.path.join(data_directory, 'wordnet_mammal_hypernyms.tsv')

from gensim.models.poincare import PoincareModel, PoincareKeyedVectors, PoincareRelations
relations = PoincareRelations(file_path=wordnet_mammal_file, delimiter='\t')
model = PoincareModel(train_data=relations, size=2, burn_in=0)
model.train(epochs=1, print_every=500)

models_directory = os.path.join(poincare_directory, 'models')
test_model_path = os.path.join(models_directory, 'gensim_model_batch_size_10_burn_in_0_epochs_50_neg_20_dim_50')
model = PoincareModel.load(test_model_path)


