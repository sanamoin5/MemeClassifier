"""
Offensive speech classification on meme dataset pre-trained on fox news dataset

0: Non-offensive speech,
1: Offensive speech
"""

# pip install transformers==3.0

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

path = '../data/MultiOFF/'
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# return split of dataset
def get_dataset(df, seed, test_size):
    return train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)


def convert_data_to_examples(train, test, data_col, label_col):
    train_InputExamples = train.apply(
        lambda x: InputExample(guid=None, text_a=x[data_col], text_b=None, label=x[label_col]), axis=1)

    validation_InputExamples = test.apply(
        lambda x: InputExample(guid=None, text_a=x[data_col], text_b=None, label=x[label_col]), axis=1)

    return train_InputExamples, validation_InputExamples


def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    input_features = []

    for e in examples:
        input_dict = tokenizer.encode_plus(e.text_a, add_special_tokens=True, max_length=max_length,
                                           return_token_type_ids=True, return_attention_mask=True,
                                           pad_to_max_length=True, truncation=True)

        input_ids, token_type_ids, attention_mask = (
            input_dict["input_ids"], input_dict["token_type_ids"], input_dict['attention_mask'])

        input_features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in input_features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


DATA_COLUMN = 'DATA_COLUMN'
LABEL_COLUMN = 'LABEL_COLUMN'


def execute_training(model, train, test, cp_callback, epoch):
    train.columns = [DATA_COLUMN, LABEL_COLUMN]
    test.columns = [DATA_COLUMN, LABEL_COLUMN]

    # create training and validation data
    train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN)

    train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
    train_data = train_data.batch(32)

    validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
    validation_data = validation_data.batch(32)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-6, epsilon=1e-08, clipnorm=1.0),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

    model.fit(train_data, epochs=epoch, validation_data=validation_data, callbacks=[cp_callback])

    print('Predicting...')
    preds = model.predict(validation_data)

    # classification report
    print(classification_report(test[LABEL_COLUMN], np.argmax(preds[0], axis=1)))


def train_fox():
    # initialize bert for 2 labels
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                            trainable=True,
                                                            num_labels=2)

    print(model.summary())

    # Initialize checkpoints
    checkpoint_path = "../weights/bert_fox/model_bert_fox_10epochs.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # Read hate dataset and convert it into train and test
    df = pd.read_csv("../data/fox_news.csv")
    df = df.drop(columns=['Unnamed: 0'])

    df['label'] = df['label'].replace({2: 1})
    train, test = get_dataset(df, 11, 0.2)

    # train model
    execute_training(model, train, test, cp_callback, 10)


# Fine tuning using meme dataset
def finetune_memetext(df_train, df_test):
    checkpoint_path = "../weights/bert_fox_meme/model_bert_fox_memetext_7epochs.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                            trainable=True,
                                                            num_labels=2)

    # load fox news pretrained weights of bert
    model.load_weights('../weights/bert_fox/model_bert_fox_10epochs.ckpt')

    # Read meme dataset and convert it into train and test
    train = df_train[['sentence', 'label']]
    test = df_test[['sentence', 'label']]

    # train model
    execute_training(model, train, test, cp_callback, 7)


def predict_model(sents_path):
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                            trainable=True,
                                                            num_labels=2)

    # load fox news and fine tuned meme pretrained weights of bert
    model.load_weights('../weights/bert_fox_meme/model_bert_fox_memetext_7epochs.ckpt')

    df_meme = pd.read_csv(sents_path)
    df_meme['label'] = df_meme['label'].replace({'offensive': 1})
    df_meme['label'] = df_meme['label'].replace({'non-offensive': 0})
    # , image_name, sentence, label : columns
    outputs = []
    for index, row in df_meme.iterrows():
        inputs = tokenizer(row['sentence'], return_tensors="tf")
        inputs["labels"] = tf.reshape(tf.constant(1), (-1, 1))
        output = model(inputs)[1]
        arg_max_out = np.argmax(output)
        outputs.append(
            [row['image_name'], row['sentence'], output, np.max(output), arg_max_out,
             'offensive' if arg_max_out == 1 else 'non-offensive', row['label'],
             'offensive' if row['label'] == 1 else 'non-offensive', 'bert_fox'])
    df_pred = pd.DataFrame(outputs,
                      columns=['file_name', 'text', 'pred_scores', 'pred_scores_max', 'pred_class_val', 'pred_class_',
                               'actual_value', ' actual_class_val', 'model'])
    df_pred.to_csv('../outputs/bert_meme_fox_test.csv')
    return df_pred


def train_model():
    # skip training if weights already present
    ckpt_file_fox = Path("../weights/bert_fox/model_bert_fox_10epochs.ckpt")
    if not ckpt_file_fox.is_file():
        with open('../weights/bert_fox/model_bert_fox_10epochs.ckpt', 'w'):
            pass
        train_fox()
    else:
        print('Weights for bert on fox already present!')

    # skip training if weights already present
    ckpt_file_meme = Path("../weights/bert_fox_meme/model_bert_fox_memetext_7epochs.ckpt")
    if not ckpt_file_meme.is_file():
        with open('../weights/bert_fox_meme/model_bert_fox_memetext_7epochs.ckpt', 'w'):
            pass

        df_meme = pd.read_csv(path + 'multioff_offensive.csv').append(pd.read_csv(path + 'multioff_non_offensive.csv'))
        df_train_meme, df_test_meme = get_dataset(df_meme, 11, 0.2)

        df_train_meme['label'] = df_train_meme['label'].replace({'offensive': 1})
        df_train_meme['label'] = df_train_meme['label'].replace({'non-offensive': 0})
        df_test_meme['label'] = df_test_meme['label'].replace({'offensive': 1})
        df_test_meme['label'] = df_test_meme['label'].replace({'non-offensive': 0})

        finetune_memetext(df_train_meme, df_test_meme)
    else:
        print('Weights for bert on fox finetuned with meme text already present!')


def run_model(predict_sents_path):
    train_model()
    return predict_model(predict_sents_path)


df = run_model('../data/multioff_test.csv')
print(df)
