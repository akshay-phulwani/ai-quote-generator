# AI Quote Generator 🤖

A Deep Learning NLP project that generates motivational quotes using an LSTM (Long Short-Term Memory) neural network. The application is built with TensorFlow and deployed using Streamlit.

## 🚀 Live Demo

**Live App:**  
https://ai-quote-generator-akshay.streamlit.app/

## 📂 GitHub Repository

**Source Code:**  
https://github.com/akshay-phulwani/ai-quote-generator

## ✨ Features

- Generate quotes from a starting word or phrase
- Next-word prediction using LSTM
- Interactive Streamlit web interface
- Trained on a quote dataset
- Deployed online

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- LSTM
- NumPy
- Pandas
- Streamlit
- Pickle

## 📊 Project Workflow

```text
Dataset
   ↓
Text Cleaning
   ↓
Tokenization
   ↓
Sequence Generation
   ↓
Padding
   ↓
LSTM Model Training
   ↓
Quote Generation
   ↓
Streamlit Deployment
```

## 📁 Project Structure

```text
ai-quote-generator/
│
├── app.py
├── quote_generator.keras
├── tokenizer.pkl
├── max_len.pkl
├── requirements.txt
├── runtime.txt
└── README.md
```

## 🧠 Model Details

- Model Type: LSTM (Long Short-Term Memory)
- Framework: TensorFlow / Keras
- Task: Next Word Prediction
- Final Training Accuracy: ~38%
- Text Generation using probabilistic sampling

## 🎯 Future Improvements

- Attention Mechanism
- Transformer-based model
- Better UI design
- Copy-to-clipboard button
- Quote categories
- Improved text generation quality

## 👨‍💻 Author

**Akshay Phulwani**

- GitHub: https://github.com/akshay-phulwani

---

⭐ If you found this project interesting, consider giving it a star on GitHub.
```
