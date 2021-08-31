# creates context using politic image classifier output, political name in text and context sentence found using cosine similarity of sentence and wikipedia data
import pandas as pd

from get_similar_sentences import *
import nltk

# from effnet_image_classifier_politics import run_model     # if you want to run model on custom dataset path


politics_text = 'Barack Obama Donald Trump Hillary Clinton George Bush Bill Clinton Bernie Sanders Joseph Robinette Biden Joe Biden Kamala Devi Harris Nancy Pelosi Michael Richard Pence Kamala Harris Mike Pence'.lower()
politics_tokens = politics_text.split(' ')

# creates context using politic image classifier output, political name in text and context sentence
def create_context(context, img_politic, txt_politic):
    context = context.lower()
    img_politic = img_politic.lower()
    txt_politic = txt_politic.lower()
    img_token = nltk.word_tokenize(img_politic)
    txt_token = nltk.word_tokenize(txt_politic)
    context_token = nltk.word_tokenize(context)

    context_img_common = ' '.join([i for i in img_token if i in context_token])
    context_txt_common = ' '.join([i for i in txt_token if i in context_token])

    if context_img_common == '' and context_txt_common == '' and img_politic != 'animated' and img_politic != 'animals':
        return img_politic + ' ' + txt_politic + ' ' + context
    elif context_img_common == '' and img_politic != 'animated' and img_politic != 'animals':
        return img_politic + ' ' + context
    elif context_txt_common == '':
        return txt_politic + ' ' + context
    else:
        return context

# loads and creates context data
def load_data(path, politic_file, outfile):
    df = pd.read_csv(path)
    df_politics = pd.read_csv(politic_file)
    new_context_sents = []
    for index, row in df.iterrows():
        politic_in_text = ' '.join([i for i in nltk.word_tokenize(row['sentence'].lower()) if i in politics_tokens])

        # find similar sentences to the one in text from wikipedia
        context_sents = ' '.join(get_similar_sentences(row['sentence'], 1))
        new_context_sents.append(
            create_context(context_sents, ' '.join(
                df_politics[df_politics['file_name'] == row['image_name']]['pred_class_'].item().split('_')),
                           politic_in_text))
    df['context_sentences'] = new_context_sents
    df.to_csv(outfile)

    return df


# df = load_data('../data/multioff_off_nonoff.csv','../outputs/effnet_meme_politics_all_train.csv', '../data/multioff_context_train_data.csv')
df = load_data('../data/multioff_test.csv', '../outputs/effnet_meme_politics_test.csv',
               '../data/multioff_context_test_data.csv')
