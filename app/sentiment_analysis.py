from app.task.taska_youtube_handling import YouTubeSentimentExtractor
from app.lancedb.db_handler import lance_db_update
from app.task.taskb_clustering import TopicModeling
import pandas as pd

def extract_sentiment_topic(product:str):
    '''
    Extracts the sentiment from the product.

    Args:
        product (str): The product to extract the sentiment from.

    Returns:
        video_link (str): The link to the video.
        positive_df (DataFrame): The positive sentiment data.
        negative_df (DataFrame): The negative sentiment data.
    
    '''
    ## Extract the sentiments from the product
    youtube_sentiment_extractor = YouTubeSentimentExtractor(product)
    sentiment_list, video_link = youtube_sentiment_extractor.run()

    ## Update the sentiments to the LanceDB
    lance_db_update(sentiment_list['positive'], "positive")
    lance_db_update(sentiment_list['negative'], "negative")

    ## Topic modeling
    topic_modeling = TopicModeling()
    positive_dict, negative_dict = topic_modeling.run(product)

    ## Structure the data
    positive_df = pd.DataFrame(positive_dict)
    negative_df = pd.DataFrame(negative_dict)

    return video_link, positive_df, negative_df

    

    

