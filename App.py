import streamlit as st
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load NLTK resources
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

# Load saved components
try:
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('stack_model_mnb_rf.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    test_accuracy = joblib.load('test_accuracy.pkl')
    y_test = joblib.load('y_test.pkl')
    y_pred = joblib.load('y_pred.pkl')
except FileNotFoundError as e:
    st.error(e)
    st.stop()

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[/(){}\[\]\|@,;]', '', text)
    text = re.sub(r'[^0-9a-z #+_]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        return {
            'J': wordnet.ADJ,
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV
        }.get(tag, wordnet.NOUN)

    tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

def plot_confusion_matrix(y_true, y_pred, label_encoder):
    cm = confusion_matrix(y_true, y_pred)
    class_names = label_encoder.classes_

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix for Stacking Ensemble Model')

    st.pyplot(fig)

# Streamlit UI
st.title("BBC News Category Classifier (Ensemble Stacked Model)")
st.write("Find out the category of BBC News articles using a Ensemble Stacked Model.")

if test_accuracy is not None:
    st.write(f"Model Accuracy: **{test_accuracy:.2f}**")

if y_test is not None and y_pred is not None:
    st.write("Predicted vs Actual Data Visualization:")
    plot_confusion_matrix(y_test, y_pred, label_encoder)

user_input = st.text_area("Please input BBC News (in English)")

if st.button("Predict"):
    if user_input:
        cleaned = preprocess(user_input)
        vec = tfidf_vectorizer.transform([cleaned])
        pred = model.predict(vec)
        label = label_encoder.inverse_transform(pred)[0]
        st.success(f"Predicted Category: **{label}**")
    else:
        st.warning("Please enter some text first.")
