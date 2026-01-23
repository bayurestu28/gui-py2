import streamlit as st
import numpy as np
import time
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

output_b = st.session_state.tweet_df1

output_b = output_b[['text', 'case_folding','filtering','tokenizing','normalization_word','stopword_removal','stemming', 'sentiment_vader_3Sentiment']]
#output_a

if "accuracy_b" not in st.session_state:
    st.session_state.accuracy_b = 0

if "precision_b" not in st.session_state:
    st.session_state.precision_b = 0

if "recall_b" not in st.session_state:
    st.session_state.recall_b = 0

if "f1_b" not in st.session_state:
    st.session_state.f1_b = 0
    
if "elapsed_time_b" not in st.session_state:
    st.session_state.elapsed_time_b = 0

if "stopped_epoch_b" not in st.session_state:
    st.session_state.stopped_epoch_b = 0

data_indo = output_b

max_features = 10000
maxlen = 20
embedding_dims = 10
tweet = data_indo.stemming.values
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(tweet)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(tweet)
padded_sequence = pad_sequences(encoded_docs, maxlen)

sentiment_label = data_indo.sentiment_vader_3Sentiment.factorize()
y = sentiment_label[0]
X = np.array(padded_sequence)
y = np.array(y)

k_fold = st.number_input('K-Fold: ')
k_fold = int(k_fold)

aktivasi = st.selectbox(
    'Silakan Pilih Fungsi aktivasi yang akan digunakan',
    ('Sigmoid', 'Tanh', 'Softmax')
)

dropout_option = st.selectbox(
    'Pilihan Dropout, Jika Ya maka nilai dropout 0,5',
    ('Tidak', 'Ya')
)

col1, col2 = st.columns(2)

# Inputan pertama di col1
with col1:
    input1 = st.text_input("Neuron 1")
    value_input1 = input1

# Inputan kedua di col2
with col2:
    input2 = st.text_input("Neuron 2")
    value_input2 = input2


button_1b, button_2b = st.columns(2)
if button_1b.button("Proses RNN"):
    kfold = KFold(n_splits=k_fold, shuffle=True, random_state=100)
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    layer_1 = int(value_input1)
    layer_2 = int(value_input2)

    model = Sequential()
    model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
    model.add(SimpleRNN(layer_1, return_sequences=True, input_shape=(X_train.shape[1], 1)))

    if dropout_option == 'Ya':
        model.add(Dropout(0.5))

    model.add(SimpleRNN(layer_2))

    if dropout_option == 'Ya':
        model.add(Dropout(0.5))

    model.add(Dense(1, activation=aktivasi.lower()))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    start_time = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=400, batch_size=64, callbacks=[early_stop])
    elapsed_time = time.time() - start_time
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    st.session_state.model_b = model
    st.session_state.accuracy_b = accuracy*100
    st.session_state.precision_b= precision*100
    st.session_state.recall_b = recall*100
    st.session_state.f1_b = f1*100
    st.session_state.elapsed_time_b = elapsed_time
    st.session_state.stopped_epoch_b= early_stop.stopped_epoch

if button_2b.button("Reset Session"):
    st.session_state.accuracy_b = 0
    st.session_state.precision_b = 0
    st.session_state.recall_b = 0
    st.session_state.f1_b = 0
    st.session_state.elapsed_time_b = 0
    st.session_state.stopped_epoch_b = 0
    if "model_b" in st.session_state:
        del st.session_state["model_b"]
    st.success("Session telah direset")

col6, col7,  = st.columns(2)
out1 = col6.metric(label="Akurasi: ", value=str(st.session_state.accuracy_b))
out2 = col7.metric(label="Precission: ", value=str(st.session_state.precision_b))

col8, col9= st.columns(2)
out3 = col8.metric(label="Recall: ", value=str(st.session_state.recall_b))
out4 = col9.metric(label="F1-Score: ", value=str(st.session_state.f1_b))

col10, col11 =  st.columns(2)
out5 = col10.metric(label="epoch: ", value=str(st.session_state.stopped_epoch_b))
out6 = col11.metric(label="Trainng Time: ", value=str(st.session_state.elapsed_time_b))


if 'model_b' not in st.session_state:
    st.warning("Lakukan training terlebih dahulu.")
else:
    st.success("Session Sudah terisi")
    model_ab = st.session_state.model_b

    model_name = st.text_input("Nama File Model:")
    
    if st.button("Save Model"):
        if model_name:
            model_path = os.path.join("save_modelling", model_name + ".h5")
            model_aa.save(model_path)
            st.success(f"Model telah disimpan dengan nama '{model_name}.h5'")
            #st.markdown(get_download_link(model_path, model_name), unsafe_allow_html=True)
        else:
            st.warning("Nama file model tidak boleh kosong.")




