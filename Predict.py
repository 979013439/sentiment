import pickle
import pandas as pd
vectorizer = pickle.load(open('./sentiment-analysis-lr-vocabulary.pkl', 'rb'))
model      = pickle.load(open('./sentiment-analysis-lr-model.pkl', 'rb'))
sentence = ["he do not like this phone"]
res=model.predict(vectorizer.transform(sentence))
if(res[0]==0):
    print('Negatvie')
elif(res[0]==1):
    print('Positive')
else:
    print('Neutral')
