# be sure to use appropriate inputs

import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import numpy as np

filename    = 'data/source_data.txt'
data        = open(filename).read()
data        = data.lower()
# Find all the unique characters
chars       = sorted(list(set(data)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
ix_to_char  = dict((i, c) for i, c in enumerate(chars))
vocab_size  = len(chars)

seq_length = 100
list_X = [ ]
list_Y = [ ]
for i in range(0, len(chars) - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	list_X.append([char_to_int[char] for char in seq_in])
	list_Y.append(char_to_int[seq_out])
n_patterns = len(dataX)

X  = np.reshape(list_X, (n_patterns, seq_length, 1)) 
# Encode output as one-hot vector
Y  = np_utils.to_categorical(list_Y)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

filename = "weights.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

start   = np.random.randint(0, len(X) - 1)
pattern = np.ravel(X[start]).tolist()


output = []
for i in range(250):
    x           = np.reshape(pattern, (1, len(pattern), 1))
    x           = x / float(vocab_size)
    prediction  = model.predict(x, verbose = 0)
    index       = np.argmax(prediction)
    result      = index
    output.append(result)
    pattern.append(index)
    pattern = pattern[1 : len(pattern)]

print ("\"", ''.join([ix_to_char[value] for value in output]), "\"")
