import os
import requests
import asyncio
from os import pathlib

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores.pgvector import PGVector
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain.chat_models import ChatOpenAI


API_KEY = os.environ["OPENAI_API_KEY"]

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

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


async def get_github_repos():
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    url = "https://api.github.com/repositories"
    await resp = requests.get(url, headers=headers)
    for data in resp.json():
        yield data["full_name"], data["html_url"]


async def read_github_repo(full_name, html_url):
    repo_url = html_url + ".git"
    with tempfile.TemporaryDirectory() as d:
        await subprocess.check_call(
            f"git clone {repo_url}",
            cwd=d,
            shell=True,
        )
        await git_sha = (
            subprocess.check_output("git rev-parse HEAD", shell=True, cwd=d)
            .decode("utf-8")
            .strip()
        )
        repo_path = pathlib.Path(d)
    
        await markdown_files = list(html_url.rglob("*.md")) + list(html_url.rglob("*.mdx"))
        
        for markdown_file in markdown_files:
            with open(markdown_file, "r") as f:
                relative_path = markdown_file.relative_to(repo_path)
                github_url = f"{html_url}/blob/{git_sha}/{relative_path}"
                yield Document(page_content=f.read(), metadata={"source": github_url, "name": full_name})


async def split_markdown(markdown_document):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    # MD splits
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    await md_header_splits = markdown_splitter.split_documents(markdown_document)
    
    # Char-level splits
    chunk_size = 1000
    chunk_overlap = 0
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    
    # Split
    await docs = text_splitter.split_documents(md_header_splits)
    return docs


async def store_embeddings(docs, collection_name):
    await embeddings = OpenAIEmbeddings()
    
    await PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        connection_string=CONNECTION_STRING,
        distance_strategy=DistanceStrategy.COSINE,
        openai_api_key=API_KEY,
        pre_delete_collection=False,
    )
    return "embeddings stored successfully!"


collection_name = "super-docs"

async def ingest_docs():
    repos = await get_github_repos()
    tasks = []
    for name, url in repos:
        # Create tasks for reading each repo docs asynchronously
        tasks.append(read_github_repo(name, url))

    # Wait for all tasks to complete
    texts = await asyncio.gather(*tasks)
    docs = await split_markdown(texts)
    await store_embeddings(docs, collection_name)


if __name__ == "__main__":
    ingest_docs()
    