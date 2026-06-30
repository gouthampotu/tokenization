import streamlit as st
from transformers import AutoTokenizer

@st.cache_resource
def get_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")

tokenizer = get_tokenizer()

def tokenize_and_get_ids(sentence):
    if not sentence:
        return [], []
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, token_ids

st.title("Sentence Tokenizer and Token ID Generator")
st.write("Enter a sentence to see its tokens and their corresponding IDs.")

user_sentence = st.text_area("Enter your sentence here:", "chatgpt is powerful")

if user_sentence:
    tokens, token_ids = tokenize_and_get_ids(user_sentence)

    st.subheader("Tokens:")
    st.write(tokens)

    st.subheader("Token IDs:")
    st.write(token_ids)
else:
    st.write("Please enter a sentence to tokenize.")
