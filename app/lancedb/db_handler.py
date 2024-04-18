from llama_index.core import Document, StorageContext, ServiceContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from app.genai.embedding_model import get_embedding_model
from app.genai.llm import get_llamaindex_llm


def lance_db_update(doc_list, sentiment):
    docs = []

    vector_store = LanceDBVectorStore(uri="./app/lancedb/tmp", table_name=sentiment)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    for row in doc_list:
        docs.append(Document(
            text=row,
            metadata={
                'sentiment': sentiment
            }))

    service_context = ServiceContext.from_defaults(
        embed_model=get_embedding_model(),
        llm=get_llamaindex_llm())

    index = VectorStoreIndex.from_documents(
        docs, 
        storage_context=storage_context, 
        service_context=service_context
        )