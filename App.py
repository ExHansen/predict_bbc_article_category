import streamlit as st
import joblib
import numpy as np
import re
import nltk
import os
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt  

# Set up NLTK data path
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Download required NLTK data with error handling
@st.cache_resource
def download_nltk_data():
    """Download and cache NLTK data"""
    downloads = [
        ('punkt', 'tokenizers/punkt'),
        ('stopwords', 'corpora/stopwords'),  # Fixed typo: was 'copora'
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
        ('punkt_tab', 'tokenizers/punkt_tab'),
        ('wordnet', 'corpora/wordnet')  # Added missing wordnet
    ]
    
    for name, path in downloads:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, download_dir=nltk_data_path)
            except Exception as e:
                st.warning(f"Could not download {name}: {e}")
                # Fallback to default NLTK download
                nltk.download(name)

# Download NLTK data
download_nltk_data()

# Load saved components with better error handling
@st.cache_resource
def load_models():
    """Load and cache all models"""
    try:
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        model = joblib.load('stack_model_mnb_rf.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        test_accuracy = joblib.load('test_accuracy.pkl')
        y_test = joblib.load('y_test.pkl')
        y_pred = joblib.load('y_pred.pkl')
        return tfidf_vectorizer, model, label_encoder, test_accuracy, y_test, y_pred
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please make sure all .pkl files are in the same directory as this app.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Load models
tfidf_vectorizer, model, label_encoder, test_accuracy, y_test, y_pred = load_models()

# Initialize NLTK components once
@st.cache_resource
def init_nltk_components():
    """Initialize NLTK components that will be reused"""
    try:
        # Test if required data is available
        stopwords.words('english')
        WordNetLemmatizer()
        PorterStemmer()
        return True
    except LookupError as e:
        st.error(f"NLTK data not properly loaded: {e}")
        return False

# Check if NLTK is properly initialized
if not init_nltk_components():
    st.stop()

# Preprocessing function with improved error handling
def preprocess(text):
    """Preprocess text with error handling"""
    try:
        # Basic cleaning
        text = text.lower()
        text = re.sub(r'[/(){}\[\]\|@,;]', '', text)
        text = re.sub(r'[^0-9a-z #+_]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Initialize tools
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        
        def get_wordnet_pos(word):
            """Map POS tag to first character used by WordNetLemmatizer"""
            try:
                tag = nltk.pos_tag([word])[0][1][0].upper()
                return {
                    'J': wordnet.ADJ,
                    'N': wordnet.NOUN,
                    'V': wordnet.VERB,
                    'R': wordnet.ADV
                }.get(tag, wordnet.NOUN)
            except:
                return wordnet.NOUN
        
        # Process tokens
        processed_tokens = []
        for token in tokens:
            if token not in stop_words and len(token) > 2:
                # Lemmatize
                pos = get_wordnet_pos(token)
                lemmatized = lemmatizer.lemmatize(token, pos)
                # Stem
                stemmed = stemmer.stem(lemmatized)
                processed_tokens.append(stemmed)
        
        return ' '.join(processed_tokens)
    
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return text.lower()  # Return basic cleaning as fallback

def plot_confusion_matrix(y_true, y_pred, label_encoder):
    """Plot confusion matrix"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        class_names = label_encoder.classes_

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix for Stacking Ensemble Model')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting confusion matrix: {e}")

# Streamlit UI
st.title("BBC News Category Classifier")
st.subheader("Ensemble Stacking Model for News Article Classification")

# Display model information
col1, col2 = st.columns(2)

with col1:
    if test_accuracy is not None:
        st.metric("Model Accuracy", f"{test_accuracy:.3f}")

with col2:
    if label_encoder is not None:
        st.metric("Number of Categories", len(label_encoder.classes_))

# Show categories
if label_encoder is not None:
    st.write("**Available Categories:**")
    st.write(", ".join(label_encoder.classes_))

# Confusion Matrix
if y_test is not None and y_pred is not None:
    st.subheader("View Model Performance (Confusion Matrix)")
    plot_confusion_matrix(y_test, y_pred, label_encoder)

# User input section
st.markdown("---")
st.subheader("Predict Article Category")

# Text input
user_input = st.text_area(
    "Enter BBC News article text:",
    placeholder="Paste your news article here...",
    height=150
)

# Prediction
if st.button("🔍 Predict Category", type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing article..."):
            try:
                # Preprocess
                cleaned = preprocess(user_input)
                
                # Vectorize
                vec = tfidf_vectorizer.transform([cleaned])
                
                # Predict
                pred = model.predict(vec)
                pred_proba = model.predict_proba(vec)[0]
                
                # Get label
                label = label_encoder.inverse_transform(pred)[0]
                
                # Display results
                st.success(f"**Predicted Category: {label}**")
                
                # Show confidence scores
                with st.expander("View Prediction Confidence"):
                    confidence_df = {
                        'Category': label_encoder.classes_,
                        'Confidence': pred_proba
                    }
                    
                    # Sort by confidence
                    sorted_indices = np.argsort(pred_proba)[::-1]
                    
                    for idx in sorted_indices:
                        category = label_encoder.classes_[idx]
                        confidence = pred_proba[idx]
                        st.write(f"**{category}**: {confidence:.3f}")
                        st.progress(confidence)
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.error("Please try again or check if the input is valid.")
    else:
        st.warning("⚠️ Please enter some text first.")

# Footer
st.markdown("---")
st.markdown("*Made by Hansen Vernandez*")