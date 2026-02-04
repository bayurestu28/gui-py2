import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import matplotlib.pyplot as plt
from functools import lru_cache

# --- NLTK resource download otomatis ---
@st.cache_resource
def download_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)

download_nltk()

# --- Inisialisasi stemmer dengan caching per kata ---
factory = StemmerFactory()
stemmer = factory.create_stemmer()

@lru_cache(maxsize=10000)
def cached_stem(word):
    return stemmer.stem(word)

# --- Inisialisasi stopword dan VADER ---
list_stopwords = set(stopwords.words('indonesian'))
sia = SentimentIntensityAnalyzer()

# Update VADER lexicon dengan file sentiwords
with open('source/_json_sentiwords_id.txt') as f:
    senti_dict = json.load(f)
sia.lexicon.update(senti_dict)

# ----------------------------
# Fungsi preprocessing
# ----------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove tabs, newline, unicode, URL, mention, hashtag
    text = re.sub(r'(@\w+|#\w+|http\S+)', ' ', text)
    text = re.sub(r'\\[tnu]', ' ', text)
    # Remove punctuation & numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = text.strip()
    return text

def tokenize(text):
    return nltk.tokenize.word_tokenize(text)

def remove_stopwords(tokens):
    return [t for t in tokens if t not in list_stopwords]

def stem_tokens(tokens):
    return [cached_stem(t) for t in tokens]

def get_compound_score(text):
    return sia.polarity_scores(text)['compound']

def label_2sent(text):
    return 'negative' if get_compound_score(text) < 0 else 'positive'

def label_3sent(text):
    score = get_compound_score(text)
    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'

# ----------------------------
# Streamlit App
# ----------------------------
def main():
    st.set_page_config(page_title="Sentiment Analysis App")
    st.markdown("<h2 style='text-align: center;'>ANALISIS SENTIMEN OPINI MASYARAKAT</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    button_1, button_2, button_3 = st.columns(3)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if button_1.button("Proses Preprocessing"):
            # Cleaning
            df['clean_text'] = df['text'].apply(clean_text)
            # Tokenizing
            df['tokens'] = df['clean_text'].apply(tokenize)
            # Stopword removal
            df['tokens_nostop'] = df['tokens'].apply(remove_stopwords)
            # Stemming
            df['stemming'] = df['tokens_nostop'].apply(stem_tokens)
            # Gabungkan menjadi string untuk sentiment
            df['stemmed_text'] = df['stemming'].apply(lambda x: ' '.join(x))
            # Sentiment
            df['compound'] = df['stemmed_text'].apply(get_compound_score)
            df['sentiment_2'] = df['stemmed_text'].apply(label_2sent)
            df['sentiment_3'] = df['stemmed_text'].apply(label_3sent)

            st.success("✅ Preprocessing dan pelabelan selesai!")
            st.session_state.df_processed = df

        if button_2.button("Tampilkan Hasil"):
            if 'df_processed' in st.session_state:
                st.write(st.session_state.df_processed)
            else:
                st.warning("Silakan proses preprocessing terlebih dahulu.")

        if button_3.button("Tampilkan Grafik"):
            if 'df_processed' in st.session_state:
                df_plot = st.session_state.df_processed
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))

                # Grafik 3 sentimen
                counts3 = df_plot['sentiment_3'].value_counts()
                axs[0].bar(counts3.index, counts3.values, color=['green','red','blue'])
                axs[0].set_title('3 Sentimen')
                for i, v in enumerate(counts3.values):
                    axs[0].text(i, v, str(v), ha='center', va='bottom')

                # Grafik 2 sentimen
                counts2 = df_plot['sentiment_2'].value_counts()
                axs[1].bar(counts2.index, counts2.values, color=['green','red'])
                axs[1].set_title('2 Sentimen')
                for i, v in enumerate(counts2.values):
                    axs[1].text(i, v, str(v), ha='center', va='bottom')

                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Silakan proses preprocessing terlebih dahulu.")
    else:
        st.info("Silakan upload file CSV terlebih dahulu.")

if __name__ == '__main__':
    main()
