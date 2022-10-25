import pickle
import pandas as pd
vectorizer = pickle.load(open('./sentiment-analysis-lr-vocabulary.pkl', 'rb'))
model      = pickle.load(open('./sentiment-analysis-lr-model.pkl', 'rb'))
sentence = "he do not like this phone"
model.predict(vectorizer.transform([sentence]))



