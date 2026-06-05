import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Documents
docs = [
    "King and queen live in a palace",
    "Artificial Intelligence is the future",
    "Virat Kohli is a cricket player",
    "Python is used for programming"
]

# Document embeddings
doc_embeddings = model.encode(docs)

st.title("Semantic Search")

query = st.text_input("Enter your query")

if st.button("Search") and query:
    # Query embedding
    query_embedding = model.encode([query])

    # Similarity scores
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    # Best match
    best_index = scores.argmax()

    st.write("### Best Match")
    st.write(docs[best_index])

    st.write("### Similarity Score")
    st.write(round(scores[best_index], 2))