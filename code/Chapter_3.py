# Installing spaCy in virtualenv

virtualenv env
source env/bin/activate
pip install spacy

# out-of-the-box: download best-matching default model
spacy download en # english model
spacy download de # german model
spacy download es # spanish model
spacy download fr # french model
spacy download xx # multi-language model
# download best-matching version of specific model for your spaCy installation
spacy download en_core_web_sm
# download exact model version (doesn't create shortcut link)
spacy download en_core_web_sm-2.0.0 --direct

# using pip to install models
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-1.2.0/en_core_web_md-1.2.0.tar.gz
# with local file
pip install /Users/you/en_core_web_md-1.2.0.tar.gz


# Using spaCy

import spacy
nlp = spacy.load('en')

doc = nlp(u'This is a sentence.')

# Using particular model

import en_core_web_md
nlp = en_core_web_md.load()
doc = nlp(u'This is a sentence.')

# Constructing tokenizer

nlp = spacy.load('en')
nlp.tokenizer = my_tokenizer

# printing POS

doc = nlp(u'John and I went to the park’')
for token in doc:
    print(token.text, token.pos_)

# printing NER

doc = nlp(u'Microsoft has offices all over Europe.')
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

# Adding stop words

my_stop_words = [u'say', u'be', u'said', u'says', u'saying']
for stopword in my_stop_words:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True

# pre-processing text

doc = nlp(u'The horse galloped down the field and past the river.')
sentence = []
for w in doc:
    # if it's not a stop word or punctuation mark, add it to our article!
    if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num:
        # we add the lematized version of the word
        sentence.append(w.lemma_)
