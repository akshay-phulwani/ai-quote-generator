import numpy as np
import pandas as pd
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split

df = pd.read_csv("qoute_dataset (1).csv")

texts = df["quote"]

texts = texts.str.lower()
texts = texts.str.replace(r"[^\w\s]", "", regex=True)
texts = texts.str.split().str.join(" ")

tok = Tokenizer(oov_token="<OOV>")
tok.fit_on_texts(texts)

vocab_size = len(tok.word_index) + 1

sequences = tok.texts_to_sequences(texts)

X = []
y = []

for s in sequences:
    for i in range(1, len(s)):
        X.append(s[:i])
        y.append(s[i])

max_len = 50

X = pad_sequences(
    X,
    maxlen=max_len,
    padding="pre",
    truncating="pre"
)

y = np.array(y)

print("Vocabulary Size :", vocab_size)
print("X Shape :", X.shape)
print("y Shape :", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=64,
        input_length=max_len
    ),

    LSTM(64),

    Dense(
        vocab_size,
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=128
)

model.save("quote_generator.keras")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tok, f)

with open("max_len.pkl", "wb") as f:
    pickle.dump(max_len, f)

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
        pred=pred[0]
        pred_word_id = np.argmax(pred)

        next_word = index_word.get(pred_word_id, "")

        seed_text += " " + next_word

    return seed_text

print(generate_quote("life", 15))
