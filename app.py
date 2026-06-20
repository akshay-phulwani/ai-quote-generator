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

def generate_quote(seed_text, next_words=5):

    seq = tok.texts_to_sequences([seed_text])

    if len(seq[0]) == 0:
        return "Please enter a different word or phrase."

    for _ in range(next_words):

        seq = tok.texts_to_sequences([seed_text])
        if len(seq[0]) == 0:
            return "Please enter another word. This word is not in the vocabulary."

        seq = seq[0]
        
        seq = pad_sequences(
            [seq],
            maxlen=max_len,
            padding="pre",
            truncating="pre"
        )

        pred = model.predict(seq, verbose=0)[0]

        pred_word_id = np.random.choice(
            len(pred),
            p=pred / np.sum(pred)
        )

        next_word = index_word.get(
            pred_word_id,
            ""
        )

        if next_word == "":
            break

        if next_word in seed_text.split():
            continue

        seed_text += " " + next_word

    return seed_text

st.set_page_config(
    page_title="AI Quote Generator",
    page_icon="🤖"
)

st.title("🤖 AI Quote Generator")

st.write(
    "Generate motivational quotes using a Deep Learning LSTM model."
)

seed_text = st.text_input(
    "Enter a starting word or sentence"
)

if st.button("Generate Quote"):

    if seed_text.strip():

        result = generate_quote(
            seed_text,
            5
        )

        st.success(result)

    else:

        st.warning(
            "Please enter some text."
        )

st.markdown("---")

st.caption(
    "Built with TensorFlow, LSTM and Streamlit"
)
