import streamlit as st
import pandas as pd
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import matplotlib.pyplot as plt
from pathlib import Path

# ===========================
# Setup: VADER Lexicon custom
# ===========================
sia = SentimentIntensityAnalyzer()
# kosongkan lexicon default
sia.lexicon.clear()

# Load lexicon custom dari file JSON
lexicon_path = Path("source/_json_sentiwords_id.txt")
with open(lexicon_path, encoding="utf-8") as f:
    senti = json.load(f)
sia.lexicon.update(senti)

# ===========================
# Streamlit App
# ===========================
def main():
    st.set_page_config(page_title="Sentiment Analysis App")
    st.markdown("<h2 style='text-align: center;'>Analisis Sentimen Opini Masyarakat Tentang Vaksin Moderna</h2>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'>F.Vivian Praska Wandita | 185314033</h6>", unsafe_allow_html=True)
    st.write("")

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    button_1, button_2, button_3 = st.columns(3)

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if button_1.button("Proses Preprocessing & Sentiment"):
            st.info("Sedang memproses data...")

            # -------------------------------
            # Case folding
            # -------------------------------
            df['case_folding'] = df['text'].str.lower()

            # -------------------------------
            # Remove special chars, links, mentions, numbers, punctuation
            # -------------------------------
            def clean_text(text):
                if not isinstance(text, str):
                    return ""
                text = re.sub(r"http\S+|www\S+|https\S+", "", text)
                text = re.sub(r"[@#]\w+", "", text)
                text = re.sub(r"\d+", "", text)
                text = text.translate(str.maketrans("", "", string.punctuation))
                text = text.strip()
                return text

            df['clean_text'] = df['case_folding'].apply(clean_text)

            # -------------------------------
            # Tokenize sederhana (split spasi)
            # -------------------------------
            df['tokens'] = df['clean_text'].apply(lambda x: x.split())

            # -------------------------------
            # Normalisasi kata (lookup Excel)
            # -------------------------------
            normalisasi_df = pd.read_excel("source/Normalisasi word.xlsx", engine='openpyxl')
            normal_dict = dict(zip(normalisasi_df.iloc[:,0], normalisasi_df.iloc[:,1]))

            df['normalized'] = df['tokens'].apply(lambda doc: [normal_dict.get(w, w) for w in doc])

            # -------------------------------
            # Stopword removal (Sastrawi / NLTK stopwords)
            # -------------------------------
            from nltk.corpus import stopwords
            stopwords_ind = set(stopwords.words('indonesian'))
            df['no_stopwords'] = df['normalized'].apply(lambda doc: [w for w in doc if w not in stopwords_ind])

            # -------------------------------
            # Stemming
            # -------------------------------
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            df['stemming'] = df['no_stopwords'].apply(lambda doc: [stemmer.stem(w) for w in doc])

            # Gabungkan kembali menjadi string untuk sentiment
            df['stemmed_text'] = df['stemming'].apply(lambda doc: " ".join(doc))

            # -------------------------------
            # Sentiment Analysis (VADER)
            # -------------------------------
            def sentiment_label_2sent(text):
                score = sia.polarity_scores(text)['compound']
                return 'positive' if score >= 0 else 'negative'

            def sentiment_label_3sent(text):
                score = sia.polarity_scores(text)['compound']
                if score > 0:
                    return 'positive'
                elif score < 0:
                    return 'negative'
                else:
                    return 'neutral'

            df['sentiment_2'] = df['stemmed_text'].apply(sentiment_label_2sent)
            df['sentiment_3'] = df['stemmed_text'].apply(sentiment_label_3sent)

            # -------------------------------
            # Simpan session state
            # -------------------------------
            st.session_state['df_processed'] = df
            st.success("Preprocessing dan Sentiment selesai!")

    else:
        if 'df_processed' in st.session_state:
            df = st.session_state['df_processed']
        else:
            df = None

    # ===========================
    # Button tampilkan hasil
    # ===========================
    if button_2.button("Tampilkan Hasil") and df is not None:
        st.write(df)

    # ===========================
    # Button tampilkan grafik
    # ===========================
    if button_3.button("Tampilkan Grafik") and df is not None:
        # 3 Sentiment
        count3 = df['sentiment_3'].value_counts()
        # 2 Sentiment
        count2 = df['sentiment_2'].value_counts()

        fig, axs = plt.subplots(1,2,figsize=(12,6))
        axs[0].bar(count3.index, count3.values, color=['green','red','blue'])
        axs[0].set_title("3 Sentimen")
        for i, v in enumerate(count3.values):
            axs[0].text(i,v,str(v),ha='center',va='bottom')

        axs[1].bar(count2.index, count2.values, color=['green','red'])
        axs[1].set_title("2 Sentimen")
        for i, v in enumerate(count2.values):
            axs[1].text(i,v,str(v),ha='center',va='bottom')

        plt.tight_layout()
        st.pyplot(fig)


if __name__ == "__main__":
    main()
