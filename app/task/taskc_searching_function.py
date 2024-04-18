from langchain_google_genai import GoogleGenerativeAIEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from app.genai.embedding_model import get_embedding_model

embed_model = get_embedding_model()

def search(query:str) -> list:
    '''
    Search similar sentences or reviews from LanceDB.

    Args:
        query (str): The query string.

    Returns:
        list: A list of similar sentences or reviews, list of sentiment, list of score.
    '''
    ## Connect to LanceDB positive and negative table
    p_vector_store = LanceDBVectorStore(uri="./app/lancedb/tmp", table_name="positive")
    n_vector_store = LanceDBVectorStore(uri="./app/lancedb/tmp", table_name="negative")

    ## Create index from LanceDB
    p_index = VectorStoreIndex.from_vector_store(p_vector_store, embed_model=embed_model)
    n_index = VectorStoreIndex.from_vector_store(n_vector_store, embed_model=embed_model)

    ## Create retriever from index
    p_retriever = p_index.as_retriever(similarity_top_k=50)
    n_retriever = n_index.as_retriever(similarity_top_k=50)

    ## Retrieve from retriever
    p_nodes = p_retriever.retrieve(query)
    n_nodes = n_retriever.retrieve(query)

    ## Concatenate positive and negative nodes, retrieve text, sentiment and score
    nodes = p_nodes + n_nodes
    nodes_text = [node.text for node in nodes]
    nodes_sentiment = [node.metadata['sentiment'] for node in nodes]
    nodes_score = [node.score for node in nodes]

    return nodes_text, nodes_sentiment, nodes_score



