!pip install chromadb

from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.vectorstores.pgvector import PGVector
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain import PromptTemplate


## PGVector needs the connection string to the database.
## We will load it from the environment variables.
import os

api_key = os.environ["OPENAI_API_KEY"]

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
    host=os.environ.get("PGVECTOR_HOST", "localhost"),
    port=int(os.environ.get("PGVECTOR_PORT", "5432")),
    database=os.environ.get("PGVECTOR_DATABASE", "postgres"),
    user=os.environ.get("PGVECTOR_USER", "postgres"),
    password=os.environ.get("PGVECTOR_PASSWORD", "postgres"),
)


## Example
# postgresql+psycopg2://username:password@localhost:5432/database_name


def get_github_repos():
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": "Bearer <YOUR-TOKEN>",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    url = "https://api.github.com/repositories"
    resp = requests.get(url, headers=headers)
    for data in resp.json():
        yield data["full_name"], data["html_url"]


def read_github_repo(full_name, html_url):
    repo_url = html_url + ".git"
    with tempfile.TemporaryDirectory() as d:
        subprocess.check_call(
            f"git clone {repo_url}",
            cwd=d,
            shell=True,
        )
        git_sha = (
            subprocess.check_output("git rev-parse HEAD", shell=True, cwd=d)
            .decode("utf-8")
            .strip()
        )
        repo_path = pathlib.Path(d)
    
        markdown_files = list(html_url.rglob("*.md")) + list(html_url.rglob("*.mdx"))
        
        for markdown_file in markdown_files:
            with open(markdown_file, "r") as f:
                relative_path = markdown_file.relative_to(repo_path)
                github_url = f"hl{html_url}/blob/{git_sha}/{relative_path}"
                yield Document(page_content=f.read(), metadata={"source": github_url, "name": full_name})


def split_markdown(markdown_document):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    # MD splits
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_documents(markdown_document)
    
    # Char-level splits
    chunk_size = 500
    chunk_overlap = 1
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    
    # Split
    docs = text_splitter.split_documents(md_header_splits)
    return docs


def store_embeddings(docs, collection_name="github_vectorstore"):
    embeddings = OpenAIEmbeddings()
    
    db = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        connection_string=connection_string,
        distance_strategy=DistanceStrategy.COSINE,
        openai_api_key=api_key,
        pre_delete_collection=False,
    )
    return db


def index_store(vector_store, documents):
    service_context = ServiceContext.from_defaults()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, 
        service_context=service_context
    )


def main():
    repos = get_github_repos
    documents = []
    for name, url in repos:
        docs = read_github_repo(name, url)
        documents.append(docs)
    chunks = split_markdown(documents)
    vectorstore = store_embeddings(chunks)
    index = index_store(vectorstore, chunks)
    return index



template="""
You are a helpful assistant who love to help people. 
Given the following sections from all documentations on github, give a comprehensive answer to the question using only the information, outputted in markdown. 
If you are unsure amd the answer is not explicitly written in the documentations, reply with some sets of clarification questions. 

Context sections:
{context}

Question:'''
{question}
'''

Answer as markdown (including related snippets if available)
"""


def query_vectorestore(query):
    store = PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=OpenAIEmbeddings(),
        collection_name="github_vectorstore",
        distance_strategy=DistanceStrategy.COSINE,
    )
    
    prompt=PromptTemplate(
        template,
        input_variables=["context", "question"],
        template_format='f-string'
    )
    
    qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.5,
        model_name="gpt-3.5-turbo"), 
        store.as_retriever(search_kwargs={"k": 20}),
        condense_question_prompt=prompt,
        chain_type="map_reduce", 
        return_source_documents=True
    )
    result = qa.run(query)
    print(result["result"])
    print(result["source_documents"])
    return result 