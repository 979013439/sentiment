from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


import re
import math

np.random.seed(1)


df_ = pd.read_csv('facebook_sentiment.csv')
classes=["positive","negative","neutral"]
df = df_[['sentimental', 'text']].copy()
# df['sentimental'].hist()
target_map = {'positive': 1, 'negative': 0, 'neutral': 2}
df['target'] = df['sentimental'].map(target_map)

# 清洗语料库
for i in range(0, df.shape[0]):
    _text = df['text'][i]
    _result: float = df['target'][i]
    if (type(_text) == float or math.isnan(_result)):
        df.drop(i, inplace=True)
        continue
    df['text'][i] = re.sub('[^a-zA-Z]', ' ', df['text'][i])


df_train, df_test, = train_test_split(df)

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
P_test = model.predict(X_test)
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
# 使用sklearn工具包中的ConfusionMatrixDisplay可视化混淆矩阵，参考plot_confusion_matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=classes)
disp.plot(
    include_values=True,            # 混淆矩阵每个单元格上显示具体数值
    cmap="viridis",                 # 不清楚啥意思，没研究，使用的sklearn中的默认值
    ax=None,                        # 同上
    xticks_rotation="horizontal",   # 同上
    values_format="f"               # 显示的数值格式
)

plt.show()


pickle.dump(vectorizer, open('./sentiment-analysis-lr-vocabulary.pkl', "wb"))
pickle.dump(model, open('./sentiment-analysis-lr-model.pkl', 'wb'))
