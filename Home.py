import streamlit as st
import pandas as pd
from PIL import Image
import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import matplotlib.pyplot as plt

# --- NLTK resource download otomatis ---
#nltk.download('punkt', quiet=True)
#nltk.download('punkt_tab', quiet=True)
#nltk.download('stopwords', quiet=True)
#nltk.download('vader_lexicon', quiet=True)

@st.cache_resource
def download_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

download_nltk()


def main():
    st.set_page_config(page_title="Sentiment Analysis App")
    st.markdown("<h2 style='text-align: center; color: black;'>ANALISIS SENTIMEN OPINI MASYARAKAT TENTANG VAKSIN MODERNA MENGGUNAKAN METODE RECURRENT NEURAL NETWORK</h2>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: black;'>F.Vivian Praska Wandita| 185314033</h6>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<h6 style='text-align: left; color: black;'>Selamat Datang !!!</h6>", unsafe_allow_html=True)

    # Upload file CSV
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    button_1, button_2, button_3 = st.columns(3)
    #button_1 = st.button("Proses preprocessing")
    #button_2 = st.button("tampilkan Grafik")

    if uploaded_file is not None:
        # Membaca file CSV
        tweet_df = pd.read_csv(uploaded_file)

        if button_1.button("Proses preprocessing"):
            # PREPROCESSING
            # Case Folding
            def lower_case(sentence):
                return sentence.lower()

            tweet_df['case_folding'] = tweet_df['text'].apply(lower_case)

            # Filtering
            def remove_tweet_special(text):
                # remove tab, new line, and back slice
                text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', "")
                # remove non ASCII (emoticon, chinese word, etc.)
                if isinstance(text, str):
                    text = text.encode('ascii', 'replace').decode('ascii')
                # remove mention, link, hashtag
                text = ' '.join(re.sub("([@_#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
                # remove incomplete URL
                return text.replace("http://", " ").replace("https://", " ")

            tweet_df['filtering'] = tweet_df['case_folding'].apply(remove_tweet_special)

            # Remove number
            def remove_number(text):
                return re.sub(r"\d+", "", text)

            tweet_df['filtering'] = tweet_df['filtering'].apply(remove_number)

            # Remove punctuation
            def remove_punctuation(text):
                return text.translate(str.maketrans("", "", string.punctuation))

            tweet_df['filtering'] = tweet_df['filtering'].apply(remove_punctuation)

            # Remove leading and trailing whitespace
            def remove_whitespace_LT(text):
                return text.strip()

            tweet_df['filtering'] = tweet_df['filtering'].apply(remove_whitespace_LT)

            tweet_df = tweet_df.drop_duplicates(subset=['filtering'])

            def tokenize(token):
                return nltk.tokenize.word_tokenize(token)

            tweet_df['tokenizing'] = tweet_df['filtering'].apply(tokenize)

            normalizad_word = pd.read_excel("source/Normalisasi word.xlsx")
            normalizad_word_dict = {}

            for index, row in normalizad_word.iterrows():
                if row[0] not in normalizad_word_dict:
                    normalizad_word_dict[row[0]] = row[1]

            def normalized_term(document):
                return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]

            tweet_df['normalization_word'] = tweet_df['tokenizing'].apply(normalized_term)

            listStopword = set(stopwords.words('indonesian'))

            def stopwords_removal(text):
                return [word for word in text if word not in listStopword]

            tweet_df['stopword_removal'] = tweet_df['normalization_word'].apply(stopwords_removal)

            # create stemmer
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()

            # stemmed
            def stemmed_wrapper(term):
                return stemmer.stem(term)

            term_dict = {}

            for document in tweet_df['stopword_removal']:
                for term in document:
                    if term not in term_dict:
                        term_dict[term] = ' '

            for term in term_dict:
                term_dict[term] = stemmed_wrapper(term)

            # apply stemmed term to data frame
            def get_stemmed_term(document):
                return [term_dict[term] for term in document]

            tweet_df['stemming'] = tweet_df['stopword_removal'].apply(get_stemmed_term)


            # nltk.download('vader_lexicon')
            # Memanfaatkan nltk VADER untuk menggunakan leksikon kustom
            sia2 = SentimentIntensityAnalyzer()
            # membersihkan leksikon VADER default
            sia2.lexicon.clear()

            # Membaca leksikon InSet
            # Leksikon InSet lexicon dibagi menjadi dua, yakni polaritas negatif dan polaritas positif;
            # kita akan menggunakan nilai compound saja untuk memberi label pada suatu kalimat

            # Membaca leksikon sentistrength_id
            with open('source/_json_sentiwords_id.txt') as f:
                data2 = f.read()

            # Mengubah leksikon sebagai dictionary

            senti = json.loads(data2)

            # Update leksikon VADER yang sudah 'dimodifikasi'
            sia2.lexicon.update(senti)


            # create stemmer
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            #tweet_df1=tweet_df
            #tweet_df1['stemming_temp']=tweet_df1['stemming'].apply(str)
            # function to clean text and do stemming
            tweet_df1 = tweet_df
            tweet_df2 = tweet_df
            tweet_df1['stemming_temp'] = tweet_df1['stemming'].apply(str)
            tweet_df2['stemming_temp'] = tweet_df2['stemming'].apply(str)
            def clean_text(text):
                # remove symbols and numbers
                text = re.sub(r'[^\w\s]+', '', text)
                text = re.sub(r'\d+', '', text)
                # do stemming
                text = stemmer.stem(text)
                return text

            # apply function to 'stemming' column and create new column 'stemming_temp'
            tweet_df1['stemming_temp'] = tweet_df1['stemming_temp'].apply(clean_text)
            tweet_df2['stemming_temp'] = tweet_df2['stemming_temp'].apply(clean_text)

            # function to get compound score from VADER
            def get_compound_score(tweet):
                return sia2.polarity_scores(tweet)['compound']

            # add 'compound_text' column to dataframe
            tweet_df1['compound_text_3Sentiment'] = tweet_df1['stemming_temp'].apply(get_compound_score)
            tweet_df2['compound_text_2Sentiment'] = tweet_df2['stemming_temp'].apply(get_compound_score)

            # melakukan pelabelan pada setiap tweet menggunakan VADER
            def label_tweet_vader_twosenti(tweet):
                compound_score_twosenti = sia2.polarity_scores(tweet)['compound']
                if compound_score_twosenti < 0:
                    return 'negative'
                else:
                    return 'positive'

            def label_tweet_vader_threesenti(tweet):
                compound_score_threesenti = sia2.polarity_scores(tweet)['compound']
                if compound_score_threesenti > 0:
                    return 'positive'
                elif compound_score_threesenti < 0:
                    return 'negative'
                else:
                    return 'neutral'


            # menambahkan kolom 'sentiment_vader' ke dalam dataframe
            tweet_df1['sentiment_vader_2Sentiment'] = tweet_df1['stemming_temp'].apply(label_tweet_vader_twosenti)
            tweet_df2['sentiment_vader_3Sentiment'] = tweet_df2['stemming_temp'].apply(label_tweet_vader_threesenti)

            # Menentukan urutan kolom yang baru
            new_column_order = ['text', 'case_folding', 'filtering', 'tokenizing', 'normalization_word', 'stopword_removal',
                    'stemming', 'stemming_temp', 'compound_text_3Sentiment', 'sentiment_vader_3Sentiment', 'compound_text_2Sentiment',
                    'sentiment_vader_2Sentiment']

            # Mengubah urutan kolom dalam DataFrame tweet_df
            
            tweet_df1 = tweet_df1.reindex(columns=new_column_order)

            st.success("Proses Preprocessing dan Pelabelan selesai!")
            

            #st.write(tweet_df)
            st.session_state.tweet_df1 = tweet_df1

    else:
        # Menggunakan tweet_df yang ada dalam session_state
        if 'tweet_df1' not in st.session_state:
            st.warning("Silakan upload file CSV terlebih dahulu.")
            
        else:
            st.success("Session Sudah terisi, Silakan pilih lihat hasil atau lihat grafik")
            tweet_df = st.session_state.tweet_df1
            

            #st.write(tweet_df)

    
    if button_2.button("tampilkan hasil "):
        tweet_df1 = st.session_state.tweet_df1
        
        st.write(tweet_df1)

    
####Button 3 (buat tampil grafik)
    if button_3.button("Tampilkan Grafik"):
        tweet_df = st.session_state.tweet_df1

         # Membuat grafik banyak label sentimen pada 3 sentimen
        sentiment_count_3Sentiment = tweet_df['sentiment_vader_3Sentiment'].value_counts()
                    # Membuat grafik banyak label sentimen pada 2 sentimen
        sentiment_count_2Sentiment = tweet_df['sentiment_vader_2Sentiment'].value_counts()

            # Membuat subplot dengan 1 baris dan 2 kolom
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            # Mengatur warna untuk setiap label sentimen
        colors_3Sentiment = ['green', 'red', 'blue']
        colors_2Sentiment = ['green', 'red']
        
                    # Menggambar grafik 3 sentimen pada subplot
        axs[0].bar(sentiment_count_3Sentiment.index, sentiment_count_3Sentiment.values, color=colors_3Sentiment)
        axs[0].set_title('Hasil Pelabelan 3 Sentimen')

            # Menampilkan angka pada batang grafik 3 sentimen
        for i, value in enumerate(sentiment_count_3Sentiment.values):
            axs[0].text(i, value, str(value), ha='center', va='bottom')


            # Menggambar grafik 2 sentimen pada subplot
        axs[1].bar(sentiment_count_2Sentiment.index, sentiment_count_2Sentiment.values, color=colors_2Sentiment)
        axs[1].set_title('Hasil Pelabelan 2 Sentimen ')

            # Menampilkan angka pada batang grafik 2 sentimen
        for i, value in enumerate(sentiment_count_2Sentiment.values):
            axs[1].text(i, value, str(value), ha='center', va='bottom')


            # Mengatur tata letak subplot agar grafik tidak tumpang tindih
        plt.tight_layout()

            # Mengkonversi plot menjadi format yang bisa ditampilkan di Streamlit
        st.pyplot(fig)

    


if __name__ == '__main__':
    main()

