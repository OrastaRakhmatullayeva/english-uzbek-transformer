import streamlit as st
from inference import translate

st.set_page_config(
    page_title="LingoUz | AI Translator",
    page_icon="🌈",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg,
        #fff1b8,
        #ffd6ec,
        #d6c8ff,
        #b9efff
    );
}

.block-container {
    max-width: 750px;
    padding-top: 3rem;
}

h1 {
    color: #6929a8 !important;
    text-align: center;
}

.stButton > button {
    background: linear-gradient(90deg, #ff4fa3, #9055ff);
    color: white;
    border: none;
    border-radius: 15px;
    font-weight: bold;
    width: 100%;
    padding: 0.8rem;
}

.stTextInput input {
    border-radius: 12px;
    border: 2px solid #d6a6ff;
}

.result {
    background: white;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    color: #6929a8;
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🌈 LingoUz")
st.markdown(
    "<h3 style='text-align:center;color:#6b48a5;'>"
    "English → Uzbek AI Word Translator ✨"
    "</h3>",
    unsafe_allow_html=True
)

st.write("Translate English words using a Transformer trained from scratch.")

word = st.text_input(
    "Enter an English word",
    placeholder="e.g. hello, book, water..."
)

if st.button("✨ Translate"):
    if word.strip():
        result = translate(word.strip().lower())

        st.markdown(
            f"<div class='result'>{result}</div>",
            unsafe_allow_html=True
        )
    else:
        st.warning("Please enter a word.")

st.markdown("---")
st.caption(
    "Built with PyTorch Transformer | "
    "100 English–Uzbek word pairs | "
    "Inspired by Attention Is All You Need (2017)"
)
