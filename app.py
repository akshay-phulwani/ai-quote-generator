import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("quote_generator.keras")

with open("tokenizer.pkl", "rb") as f:
    tok = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

index_word = tok.index_word


def generate_quote(seed_text, next_words=15):

    for _ in range(next_words):

        seq = tok.texts_to_sequences([seed_text])[0]

        seq = pad_sequences(
            [seq],
            maxlen=max_len,
            padding="pre",
            truncating="pre"
        )

        pred = model.predict(seq, verbose=0)

        pred_word_id = np.argmax(pred)

        next_word = index_word.get(pred_word_id, "")

        seed_text += " " + next_word

    return seed_text


st.title("AI Quote Generator")

seed_text = st.text_input(
    "Enter starting word or sentence"
)

if st.button("Generate Quote"):

    if seed_text:

        result = generate_quote(
            seed_text,
            15
        )

        st.success(result)

    else:

        st.warning(
            "Please enter some text"
        )