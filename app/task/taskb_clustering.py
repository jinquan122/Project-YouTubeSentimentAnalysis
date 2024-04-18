from google.api_core import retry
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity
import lancedb
from app.genai.llm import get_gemini
from pandas import DataFrame
from app.genai.prompt import get_topic_prompt


class TopicModeling():
    def __init__(self):
        self.model = get_gemini()

    def _get_embeddings(self, sentiment:str) -> DataFrame:
        '''
        Get embeddings from lancedb.

        Args:
            sentiment (str): The sentiment.

        Returns:
            pandas.DataFrame: The embeddings and text.
        '''
        db = lancedb.connect('./app/lancedb/tmp')
        tbl = db.open_table(sentiment)
        df = tbl.to_pandas()
        return df
    
    def _clustering(self, df:DataFrame) -> DataFrame:
        '''
        Clustering the embeddings.

        Args:
            df (pandas.DataFrame): The embeddings and text.

        Returns:
            pandas.DataFrame: The embeddings and text and cluster.
        '''
        ## Calculate the similarity
        similarity = cosine_similarity(np.array(df['vector']).tolist())
        linkage_matrix = linkage(similarity, method='weighted')

        ## Set the threshold distance
        threshold_distance = 0.5

        ## Cluster the embeddings
        cluster = fcluster(linkage_matrix, threshold_distance, criterion='distance')
        df['cluster'] = cluster
        return df
    
    @retry.Retry(timeout=300.0)
    def _top_modeling(self, sentiment:str, product:str):
        '''
        Top modeling.

        Args:
            sentiment (str): The sentiment.
            product (str): The product.

        Returns:
            dict: The structured positive and negative sentiment.
            result: {'topic': topic, 'list': sentiments_list, 'count': len(sentiments_list), 'sentiment': sentiment}
        '''
        ## Get the embeddings and text
        df = self._get_embeddings(sentiment)

        ## Cluster the embeddings
        df = self._clustering(df)
        result = []
        others = []

        ## Get the topic
        unique_clusters = df['cluster'].unique()
        for cluster in unique_clusters:
            cluster_df = df[df['cluster'] == cluster]

            ## Get the topic with length more than 1 and group topic length 1 together named as others
            if len(cluster_df) > 1:
                sentiments_list = cluster_df['text'].to_list()
                response = self.model.generate_content(get_topic_prompt(product, sentiments_list))
                result.append({'topic':response.text, 'list': sentiments_list, 'count': len(sentiments_list), 'sentiment': sentiment})
            else:
                sentiments_list = cluster_df['text'].to_list()[0]
                others.append(sentiments_list)
        
        ## Append the others into result list
        result.append({'topic':f"others - {sentiment}", 'list': others, 'count': len(others), 'sentiment': sentiment})
        return result
            
            
    def run(self, product:str) -> dict:
        '''
        Run the topic modeling.

        Args:
            product (str): The product.

        Returns:
            dict: The structured positive and negative sentiment.
            positive_dict: {'topic': topic, 'list': sentiments_list, 'count': len(sentiments_list), 'sentiment': sentiment}
            negative_dict: {'topic': topic, 'list': sentiments_list, 'count': len(sentiments_list), 'sentiment': sentiment}
        '''
        positive_dict = self._top_modeling('positive', product)
        negative_dict = self._top_modeling('negative', product)
        
        return positive_dict, negative_dict


