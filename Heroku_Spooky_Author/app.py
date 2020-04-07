import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile , chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import LatentDirichletAllocation as LDA
import regex as re
import pickle
import nltk
from nltk.corpus import wordnet,stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from flask import Flask, request, jsonify, render_template

STOPWORDS = set(stopwords.words('english'))
lem=WordNetLemmatizer()
tf = TfidfVectorizer()

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

#Function that pipelines and prepares data for Prediction Models
def pipeline(text):
    text = text.rstrip()
    stop_free = ' '.join([word for word in text.lower().split() if ((word not in STOPWORDS))])
    punc_free=re.sub('[^a-zA-Z]', " ", str(stop_free))
    text = ' '.join(lem.lemmatize(word, get_wordnet_pos(word)) for word in word_tokenize(punc_free))
    return text

app = Flask(__name__)
NB = pickle.load(open('nb.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    text = request.form.values()
    pipe = pipeline(text)
    vec = tf.fit_transform(pipe)
    prediction = NB.predict(vec)
    lda = LDA(n_components=1, n_jobs=-1)
    lda.fit(vec)
    words = tf.get_feature_names()
    for topic_idx, topic in enumerate(lda.components_):
        TopStr = " ".join([words[i] for i in topic.argsort()[:-5 - 1:-1]])
    op1 = prediction
    op2 = TopStr 

    return render_template('index.html', prediction_text='Author: {} Topics: {}'.format(op1,op2))


if __name__ == "__main__":
    app.run(debug=True)