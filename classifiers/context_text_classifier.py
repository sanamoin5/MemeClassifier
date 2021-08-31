# classifier using parallel Bidirectional LSTM to predict offensive/non-offensive text using the context of the meme
# and the meme text

from pathlib import Path
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, concatenate
from keras.models import Model
from keras.layers import Bidirectional
from sklearn.metrics import classification_report
import pandas as pd
import re
import nltk
import tensorflow as tf
import tensorflow_hub as hub

max_len = 50


# clean text by removing unncessary keywords and applying text filters
def clean_text(text):
    text = re.sub(r'https?://\S+|www.\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if not word in set(nltk.corpus.stopwords.words('english'))]
    text = ' '.join(text)
    return text


# load data for trainig/testing
def load_data(filepath, is_not_pred=True):
    df = pd.read_csv(filepath)
    if is_not_pred:
        df = df.sample(frac=1).reset_index(drop=True)

    # replace label values to 0 and 1
    df['label'] = df['label'].replace({'offensive': 1})
    df['label'] = df['label'].replace({'non-offensive': 0})
    y = df['label'].values

    # clean text
    df['sentence'] = df['sentence'].apply(lambda x: clean_text(x))
    df['context_sentences'] = df['context_sentences'].apply(lambda x: clean_text(x))

    x = df['sentence'].values
    x_context = df['context_sentences'].values

    # tokenize text and convert to sequences
    tokenizer = Tokenizer(num_words=100)
    tokenizer.fit_on_texts(x)
    x = tokenizer.texts_to_sequences(x)

    tokenizer = Tokenizer(num_words=100)
    tokenizer.fit_on_texts(x_context)
    x_context = tokenizer.texts_to_sequences(x_context)

    vocab_size = len(tokenizer.word_index) + 1

    # pad sequences with zeros
    x_train = pad_sequences(x, padding='post', maxlen=max_len)
    x_context = pad_sequences(x_context, padding='post', maxlen=max_len)

    return vocab_size, x_train, x_context, y


# trains the model with the file path csv given as parameter and stores the model
def train_model(filepath):
    vocab_size, x_train, x_context_train, y_train = load_data(filepath)

    embedding_dim = 50

    # create two parallel Bi-dir LSTM layers

    # Bi-dire LSTM layer for meme sentence
    model1 = Sequential()
    model1.add(layers.Embedding(input_dim=vocab_size,
                                output_dim=embedding_dim,
                                input_length=max_len))

    model1.add(Bidirectional(layers.LSTM(units=50, go_backwards=True, return_sequences=True)))
    model1.add(Bidirectional(layers.LSTM(units=10, go_backwards=True)))
    model1.add(layers.Dropout(0.5))
    model1.add(layers.Dense(8))

    # Bi-dir LSTM layer for context sentence
    model2 = Sequential()
    model2.add(layers.Embedding(input_dim=vocab_size,
                                output_dim=embedding_dim,
                                input_length=max_len))
    model2.add(Bidirectional(layers.LSTM(units=50, go_backwards=True, return_sequences=True)))
    model2.add(Bidirectional(layers.LSTM(units=10, go_backwards=True)))
    model2.add(layers.Dropout(0.5))
    model2.add(layers.Dense(8))

    # concatenating outputs of the two LSTM parallel layers and adding dense layer to them for output
    merge_one = concatenate([model1.output, model2.output])
    merge_output = Dense(8, activation='relu')(merge_one)
    output = Dense(1, activation='sigmoid')(merge_output)
    model = Model(inputs=[model1.input, model2.input], outputs=output)

    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=['accuracy'])

    # provide two input data for the two LSTM layers
    model.fit([x_train, x_context_train], y_train, epochs=7, batch_size=4, verbose=1)

    model.save('../weights/context_classifier_text_7epochs.h5')


# predict the model given the prediction csv filepath
def predict_model(filepath):
    # laod model
    model = tf.keras.models.load_model('../weights/context_classifier_text_7epochs.h5',
                                       custom_objects={'KerasLayer': hub.KerasLayer})
    _, x_test, x_context_test, y_test = load_data(filepath)

    # perform prediction
    preds = model.predict([x_test, x_context_test])

    # set threshold of 0.5 for getting 0 or 1 from sigmoid layer
    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0

    print(pd.DataFrame(classification_report(y_test, preds, output_dict=True)).transpose())

    # store the results in output folder
    df_meme = pd.read_csv(filepath)
    # , image_name, sentence, label : columns
    outputs = []
    for index, row in df_meme.iterrows():
        outputs.append(
            [row['image_name'], row['sentence'], int(preds[index][0]),
             'offensive' if preds[index] == 1 else 'non-offensive', row['label'],
             '1' if row['label'] == 'offensive' else 0, 'context_classifier'])
    df_pred = pd.DataFrame(outputs,
                           columns=['file_name', 'text', 'pred_class_val',
                                    'pred_class_',
                                    'actual_value', ' actual_class_val', 'model'])
    df_pred.to_csv('../outputs/context_clssifier_test.csv')

    return df_pred


# run the context classififer model for training and predicting
def run_model(train_path='../data/multioff_context_train_data.csv',
              predict_path='../data/multioff_context_test_data.csv'):
    # skip training if weights already present
    ckpt_file = Path("../weights/context_classifier_text_7epochs.h5")
    if not ckpt_file.is_file():
        train_model(train_path)
    else:
        print('Weights for context classifier on multioff already present!')

    return predict_model(predict_path)


df = run_model('../data/multioff_context_train_data.csv', '../data/multioff_context_test_data.csv')
