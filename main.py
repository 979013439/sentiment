import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import pickle

np.random.seed(1)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

df_ = pd.read_csv('facebook_sentiment.csv')
df_.head()
df = df_[['sentimental', 'text']].copy()
df.head()
df['sentimental'].hist()
target_map = {'positive': 1, 'negative': 0, 'neutral': 2}
df['target'] = df['sentimental'].map(target_map)
df.head()
df_train, df_test, = train_test_split(df)
df_train.head()
vectorizer = TfidfVectorizer(max_features=2000)
X_train = vectorizer.fit_transform(df_train['text'])
X_test = vectorizer.transform(df_test['text'])
X_train.shape

Y_train = df_train['target']
Y_test = df_test['target']

model = LogisticRegression(max_iter=500)
model.fit(X_train, Y_train)
print('Train acc: ', model.score(X_train, Y_train))
print("Test  acc: ", model.score(X_test, Y_test))
Pr_train = model.predict_proba(X_train)
Pr_test = model.predict_proba(X_test)
print("Train AUC: ", roc_auc_score(Y_train, Pr_train, multi_class='ovo'))
print("Test  AUC: ", roc_auc_score(Y_test, Pr_test, multi_class='ovo'))
P_train = model.predict(X_train)
P_test  = model.predict(X_test)
cm = confusion_matrix(Y_train, P_train, normalize='true')
cm
def plot_cm(cm):
    classes = ['negative', 'positive', 'neutral']
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    ax = sn.heatmap(df_cm, annot=True, fmt='g')
    ax.set_xlabel('Predicted')
    ax.set_ylabel("Target")

plot_cm(cm)
cm_test = confusion_matrix(Y_test, P_test, normalize='true')
plot_cm(cm_test)

pickle.dump(vectorizer, open('./sentiment-analysis-lr-vocabulary.pkl', "wb"))
pickle.dump(model, open('./sentiment-analysis-lr-model.pkl', 'wb'))
