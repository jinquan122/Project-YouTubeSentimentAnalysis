from typing import List
import wikipedia
from streamlit_searchbox import st_searchbox
import streamlit as st
import lancedb
import pandas as pd
from pandas import DataFrame
from wordcloud import WordCloud


def _wikipedia_function(searchterm: str) -> List[any]:
    return wikipedia.search(searchterm) if searchterm else []

def wikipedia_search() -> str:
    '''
    Wikisearch.

    Returns:
        Product (str): Product name.
    '''
    product = st_searchbox(
        _wikipedia_function,
        key="wiki_searchbox",
        edit_after_submit=True,
    )
    return product

def drop_lance_db() -> None:
    '''
    Drop lancedb.
    '''
    db = lancedb.connect('./app/lancedb/tmp')
    if all(item in db.table_names() for item in ['positive', 'negative']):
        db.drop_table('positive')
        db.drop_table('negative')

def build_sentiment_dashboard(positive_df: DataFrame, negative_df:DataFrame) -> None:
    '''
    Build sentiment dashboard.

    Args:
        positive_df (pandas.DataFrame): Positive sentiment dataframe.
        negative_df (pandas.DataFrame): Negative sentiment dataframe.
    '''
    total_positive_count = positive_df['count'].sum()
    total_negative_count = negative_df['count'].sum()
    total_count = total_positive_count + total_negative_count
    positive_percent = round((total_positive_count/total_count)*100)
    negative_percent = round((total_negative_count/total_count)*100)

    cola, colb = st.columns(2)
    with cola:
        container = st.container(border=True)
        container.write("Positive Sentiment Percentage")
        container.subheader(f"{positive_percent}%")
    with colb:
        container = st.container(border=True)
        container.write("Negative Sentiment Percentage")
        container.subheader(f"{negative_percent}%")

def build_barchart(positive_df:DataFrame, negative_df:DataFrame) -> None:
    '''
    Build barchart.

    Args:
        positive_df (pandas.DataFrame): Positive sentiment dataframe.
        negative_df (pandas.DataFrame): Negative sentiment dataframe.
    '''
    sentiment_df = pd.concat([positive_df, negative_df])
    sentiment_df = sentiment_df[sentiment_df['topic'] != 'others - positive']
    sentiment_df = sentiment_df[sentiment_df['topic'] != 'others - negative']
    sentiment_df = sentiment_df[['topic','count','sentiment']]
    sorted_df = sentiment_df.sort_values(by=['count'], ascending=False)
    sorted_df = sorted_df.reset_index(drop=True)
    sorted_df = sorted_df.head(20)
    st.bar_chart(sorted_df, x = "topic", y = "count", color = "sentiment", use_container_width=True)

def build_wordcloud(positive_df:DataFrame, negative_df:DataFrame) -> None:
    '''
    Build wordcloud.

    Args:
        positive_df (pandas.DataFrame): Positive sentiment dataframe.
        negative_df (pandas.DataFrame): Negative sentiment dataframe.
    '''
    colA, colB = st.columns(2)
    with colA:
        all_text = ' '.join(' '.join(text_list) for text_list in positive_df['list'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        st.subheader("Positive Word Cloud")
        st.image(wordcloud.to_array(), caption='Word Cloud', use_column_width=True)
    with colB:
        all_text = ' '.join(' '.join(text_list) for text_list in negative_df['list'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        st.subheader("Negative Word Cloud")
        st.image(wordcloud.to_array(), caption='Word Cloud', use_column_width=True)

def build_sentiment_point(positive_df:DataFrame, negative_df:DataFrame) -> None:
    '''
    Build sentiment point.

    Args:
        positive_df (pandas.DataFrame): Positive sentiment dataframe.
        negative_df (pandas.DataFrame): Negative sentiment dataframe.
    '''
    col1, col2 = st.columns(2)
    with col1:
        st.header("Positive sentiment")
        for i, row in positive_df.iterrows():
            with st.popover(f"{row['topic']}"):
                st.markdown("Points ⚽:")
                for point in row['list']:
                    st.write(f"- {point}")
    with col2:
        st.header("Negative sentiment")
        for i, row in negative_df.iterrows():
            with st.popover(f"{row['topic']}"):
                st.markdown("Points ⚽:")
                for point in row['list']:
                    st.write(f"- {point}")

def build_video_link(video_link:list) -> None:
    '''
    Build video link.

    Args:
        video_link (list): Video link.
    '''
    col1, col2, col3 = st.columns(3)
    column_list = [col1, col2, col3]
    for i, video in enumerate(video_link):
        col = column_list[i % len(column_list)]
        with col:
            st.image(video['thumbnail'])
            st.link_button("Go to Video", video['link'])