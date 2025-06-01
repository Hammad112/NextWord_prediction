import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# --- Prediction functions ---
def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

def predict_next_word(seed_text, model, tokenizer, max_sequence_len, num_words=1, temperature=1.0):
    output_text = seed_text
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([output_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        preds = model.predict(token_list, verbose=0)[0]
        predicted_word_index = sample_with_temperature(preds, temperature)
        
        # Map index to word
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_text += ' ' + word
                break
    return output_text

# --- Streamlit App ---
st.set_page_config(page_title="Word Predictor", page_icon="🧠", layout="centered")

st.markdown("## 🧠 AI Word Predictor")
st.markdown("Type a sentence and watch AI predict the next word(s). Adjust creativity using the temperature slider.")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
num_words = st.sidebar.slider("Number of Words to Predict", 1, 10, 3)
temperature = st.sidebar.slider("Creativity (Temperature)", 0.2, 1.5, 1.0, step=0.1)

# Load model and tokenizer (placeholder paths)
@st.cache_resource
def load_resources():
    # Load your actual model and tokenizer here
    model = load_model('model.h5')
    import pickle
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    max_sequence_len = model.input_shape[1]
    return model, tokenizer, max_sequence_len

model, tokenizer, max_sequence_len = load_resources()

# Input text
seed_text = st.text_input("✏️ Enter seed text:", "Once upon a time")

if st.button("🔮 Predict"):
    with st.spinner("Thinking..."):
        result = predict_next_word(seed_text, model, tokenizer, max_sequence_len, num_words, temperature)
    st.success("Prediction complete!")
    st.markdown(f"### 📝 Result:\n\n**{result}**")

st.markdown("---")

