from flask import Flask,request, url_for, redirect, render_template, jsonify
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import load_model
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pandas as pd
import numpy as np
import pickle
import os


nltk.download('stopwords')
nltk.download('punkt')

maxlen = 20   

app = Flask(__name__)



def loadModels(model_path, encoder_path):
	model_path = os.path.join(model_path, "bi_model.h5")
	encoder_path = os.path.join(encoder_path, "tokenizer.h5")
	model = load_model(model_path)
	with open(encoder_path, 'rb') as pickle_file:
		encoder = pickle.load(pickle_file)
	return model, encoder

def preprocess(par):
	X = []
	stop_words = set(nltk.corpus.stopwords.words("english"))
	tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
	tmp = []
	sentences = nltk.sent_tokenize(par)
	for sent in sentences:
		sent = sent.lower()
		tokens = tokenizer.tokenize(sent)
		filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
		tmp.extend(filtered_words)
		#X.append(tmp)
	return tmp


def transform(X, maxlen):
	#X = preprocess(txt)
	tmp = np.array(X)
	tmp = tmp.reshape(1, tmp.shape[0])
	X = encoder.texts_to_sequences(tmp.tolist())
	return pad_sequences(X, maxlen)


def predict_news(txt, maxlen, clf_model, txt_encoder):
	X = preprocess(txt)
	X = transform(X, maxlen)
	y = clf_model.predict(X)
	return y

model, encoder = loadModels('models', 'models')
   

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		my_prediction = predict_news(message,maxlen,model,encoder)
	return render_template('result.html', prediction=my_prediction)



if __name__ == '__main__':
	app.run(debug=True)