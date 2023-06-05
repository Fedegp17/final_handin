#Data cleaning and preprocessing

def read_file(filepath):
    with open(filepath, 'r') as f:
        return f.read()

#read_file('Avengers.Endgame.txt')
#print(read_file('Avengers.Endgame.txt'))

import spacy
nlp = spacy.load("en_core_web_sm",disable=['parser', 'tagger', 'ner'])
doc = nlp(read_file('Avengers.Endgame.txt'))
#print(doc)

nlp.max_length = 1198623

def separate_punct(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']

d = read_file('Avengers.Endgame.txt')
tokens = separate_punct(d)

print(len(tokens))

train_len = 25+1
text_sequences = []

for i in range(train_len, len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequences.append(seq)

print(text_sequences[1])

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences)

print(sequences[1])
for i in sequences[1]:
    print(f'{i} : {tokenizer.index_word[i]}')

print(tokenizer.index_word)
print(tokenizer.word_counts)

vocabulary_size = len(tokenizer.word_counts)
print(vocabulary_size)

#Split the sequences into X and y

import numpy as np
sequences = np.array(sequences)
print(sequences.shape)

from keras.utils import to_categorical
X = sequences[:,:-1]
y = sequences[:,-1]

y = to_categorical(y, num_classes=vocabulary_size+1)
seq_len = X.shape[1]
print(seq_len)

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding

def create_model(vocabulary_size, seq_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(vocabulary_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

model = create_model(vocabulary_size+1, seq_len)

from pickle import dump, load

#model.fit(X, y, batch_size=128, epochs=1000, verbose=1)

#model.save('my_avengers_model_reallybig.h5')

#dump(tokenizer, open('my_simpletokenizer', 'wb'))

from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words):
    output_text = []
    input_text = seed_text
    for i in range(num_gen_words):

        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
        pred_probs = model.predict(pad_encoded, verbose=0)
        pred_word_ind = np.argmax(pred_probs[0])
        pred_word = tokenizer.index_word[pred_word_ind]
        input_text += ' '+pred_word
        output_text.append(pred_word)

    return ' '.join(output_text)

import random

random.seed(42)
random_pick = random.randint(0, len(text_sequences))

random_seed_text = text_sequences[random_pick]

seed_text = ' '.join(random_seed_text)


from keras.models import load_model

model = load_model('my_avengers_model_reallybig.h5')

while True:
    seed_text = input('Enter your line: ')
    print(generate_text(model, tokenizer, seq_len, seed_text=seed_text, num_gen_words=25))
    if seed_text == 'quit':
        break