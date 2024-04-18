import streamlit as st
from app.task.taskc_searching_function import search

def main():
    st.title("Precise Sentiment Database Searching")
    st.caption("This app allows you to search the specific reviews for a specific query. You may set your prefered similarity threshold at the side bar. (Preferably 0.5)")
    ## Text input for performing search in LanceDB
    text = st.text_input("Enter a search query...")

    ## Similarity threshold slider
    with st.sidebar:
        similarity_threshold = st.number_input('Choose your similarity threshold (0-1):', min_value=0.0, max_value=1.0, step=0.01, value=0.5)
        st.write("Similarity threshold:", similarity_threshold)
        st.info("Note: The higher the similarity threshold, the less nodes will be returned.")
    if text:
        with st.spinner("Searching..."):
            ## Perform search and return node texts, node sentiments and node scores
            node_texts, node_sentiments, node_scores = search(text)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Positive sentiment")
            with col2:
                st.subheader("Negative sentiment")
                
            ## Print node texts and node sentiments with score > 0.5
            for text, sentiment, score in zip(node_texts, node_sentiments, node_scores):
                if sentiment == "positive" and score > similarity_threshold:
                    with col1:
                        st.write("-",text)
                if sentiment == "negative" and score > similarity_threshold:
                    with col2:
                        st.write("-",text)
    
if __name__ == "__main__":
    main()