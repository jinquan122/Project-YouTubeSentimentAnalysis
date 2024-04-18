from langchain_google_genai import GoogleGenerativeAIEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

def get_embedding_model():
    lc_embed_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key='AIzaSyCCcHT_A9GdjtUNyd8HF6Bl9VDsR4aYITw',
        task_type="clustering"
    )
    embed_model = LangchainEmbedding(lc_embed_model)
    return embed_model