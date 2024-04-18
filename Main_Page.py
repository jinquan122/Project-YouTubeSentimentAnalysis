import streamlit as st
from app.streamlit_helpers import wikipedia_search, drop_lance_db, build_sentiment_dashboard, build_barchart, build_wordcloud, build_sentiment_point, build_video_link
from app.sentiment_analysis import extract_sentiment_topic


def main():
    ## Set Page Info, title and subheader
    st.set_page_config(
        page_title="Youtube Sentiment Analysis", 
        page_icon="🎬", 
        layout="centered", 
        initial_sidebar_state="auto", 
        menu_items=None)
    st.title("🎬 Youtube Sentiment Analysis")
    st.subheader("This app analyzes the sentiment of a given text.")

    ## Set session state
    if 'subject' not in st.session_state:
        st.session_state['subject'] = None
    if 'video_link' not in st.session_state:
        st.session_state['video_link'] = None
    
    ## Set sidebar
    with st.sidebar:
        st.markdown("Google Gemini Pro")
        st.info("This apps is built with Google Gemini Pro. A product sentiment analysis platform for professional.")

    ## Define wikepedia search function
    product = wikipedia_search()
    st.session_state['subject'] = product

    ## Run sentiment analysis after click Run button
    if st.button("Run"):
        st.header(st.session_state['subject'] + " Sentiment Analysis Report")

        ## Clear previous data stored in LanceDB
        drop_lance_db()

        ## Extract sentiment topic
        with st.spinner("Baking sentiment in progress..."):
            video_link, positive_df, negative_df = extract_sentiment_topic(st.session_state['subject'])

        ## Build dashboard, barchart, wordcloud
        build_sentiment_dashboard(positive_df, negative_df)
        st.divider()
        build_barchart(positive_df, negative_df)
        st.divider()
        build_wordcloud(positive_df, negative_df)

        ## Build sentiment point
        with st.status("Baking sentiment point in progress..."):
            build_sentiment_point(positive_df, negative_df)
            st.success("Baking complete!")
            
        ## Build video link to Youtube
        with st.status("Baking video in progress..."):
            build_video_link(video_link)
            st.success("Baking complete!")

        ## Set session state
        st.session_state['video_link'] = video_link
        st.session_state['positive_df'] = positive_df
        st.session_state['negative_df'] = negative_df

    ## Display dashboard, barchart, wordcloud after first run
    elif st.session_state['video_link'] != None:
        st.header(st.session_state['subject'] + " Sentiment Analysis Report")
        video_link = st.session_state['video_link']
        positive_df = st.session_state['positive_df']
        negative_df = st.session_state['negative_df']

        build_sentiment_dashboard(positive_df, negative_df)
        st.divider()
        build_barchart(positive_df, negative_df)
        st.divider()
        build_wordcloud(positive_df, negative_df)

        with st.status("Baking sentiment point in progress..."):
            build_sentiment_point(positive_df, negative_df)
            st.success("Baking complete!")
            
        with st.status("Baking video in progress..."):
            build_video_link(video_link)
            st.success("Baking complete!")


if __name__ == "__main__":
    main()