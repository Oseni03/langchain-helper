from datasets import load_dataset 

data = load_dataset("squad", split="train")

# Initialize embeddings models 
from langchain.embeddings.openai import OpenAIEmbeddings 

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
model_name = "text-embedding-ada-002"

embed = OpenAIEmbeddings(
    model_name=model_name,
    openai_api_key=OPENAI_API_KEY
)

import pinecone 

YOUR_API_KEY = os.environ.get("YOUR_API_KEY")
environment = os.environ.get("PINECONE_ENVIRONMENT")

index_name = "langchain_retriever_agent"

pinecone.init(
    api_key=YOUR_API_KEY,
    environment=environment
)

if index not in pinecone.list_indexes():
    # We create a new index 
    pinecone.create_index(
        name=index_name,
        metric="cosine",
        dimension=1536 # 1536 dim of text-embedding-ada-002
    )

# connect to index
index =pinecone.GRPCIndex(index_name)
index.describe_index_stats()

