# different ML algorithms to classify offensive/non-offensive data on meme text

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

tfidf_vect = TfidfVectorizer(max_features=5000)


def load_data(filepath, text_col, label_col, is_train=True):
    # read file
    df = pd.read_csv(filepath)

    # data preprocessing
    df[text_col].dropna(inplace=True)
    df[text_col] = [entry.lower() for entry in df[text_col]]

    # shuffle
    if is_train:
        df = df.sample(frac=1).reset_index(drop=True)

    # encode label values into 0 and 1
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[label_col])

    return df[text_col], y


# k nearest neighbour algo
def knn_train(X_train, y_train):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', KNeighborsClassifier()),
                         ])

    text_clf.fit(X_train, y_train)

    return text_clf


# decision tree algo
def decision_tree_train(X_train, y_train):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', tree.DecisionTreeClassifier()),
                         ])

    text_clf.fit(X_train, y_train)

    return text_clf


# naive bayes algo
def nb_train(X_train, y_train):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         ])

    text_clf.fit(X_train, y_train)

    return text_clf


# support vector classifier
def svc_train(X_train, y_train):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LinearSVC()),
                         ])

    text_clf.fit(X_train, y_train)
    return text_clf


# train all ML models
def train_model(model_name, train_x, train_y):
    # train using naive bayes classifier
    if model_name == 'nb':
        return nb_train(train_x, train_y)
    # train using svc classifier
    elif model_name == 'svc':
        return svc_train(train_x, train_y)
    # train using k nearest neighbour classifier
    elif model_name == 'knn':
        return knn_train(train_x, train_y)
    # train using decision tree classifier
    elif model_name == 'dt':
        return decision_tree_train(train_x, train_y)


# predict according to model provided
def predict_model(model, model_name, path, sent, label):
    test_x, test_y = load_data(path, sent, label, False)
    predictions = model.predict(test_x)

    print('Model-- ', model)

    print("Accuracy Score : ", accuracy_score(predictions, test_y) * 100)

    print('Classification report: ')
    print(metrics.classification_report(test_y, predictions))

    # read test csv and create df to store results
    df = pd.read_csv(path)
    outputs = []
    for index, row in df.iterrows():
        outputs.append(
            [row['image_name'], row['sentence'], predictions[index],
             'offensive' if predictions[index] == 1 else 'non-offensive', row['label'], test_y[index], str(model)])
    df_pred = pd.DataFrame(outputs,
                           columns=['file_name', 'text', 'pred_class_val', 'pred_class_',
                                    'actual_value', ' actual_class_val', 'model'])

    # return df for predictions and store in outputs directory
    df_pred.to_csv('../outputs/' + model_name + '_multioff_fox.csv')
    return df_pred


# run model by algo name, nb for naive bayes and svm for support vector machines
def run_model(train_path, test_path, model):
    train_x, train_y = load_data(train_path, 'sentence', 'label')

    model_nb = train_model(model, train_x, train_y)

    return predict_model(model_nb, model, test_path, 'sentence', 'label')


run_model('../data/multioff_fox_combined.csv', '../data/multioff_test.csv', 'nb')
run_model('../data/multioff_fox_combined.csv', '../data/multioff_test.csv', 'svc')
run_model('../data/multioff_fox_combined.csv', '../data/multioff_test.csv', 'knn')
run_model('../data/multioff_fox_combined.csv', '../data/multioff_test.csv', 'dt')
