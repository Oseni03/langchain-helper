import requests
import pathlib
import subprocess
import tempfile


from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# Prepare Data
def get_github_docs(repo_url):
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
        
        # loader = DirectoryLoader('../', glob="**/*.md", use_multithreading=True, loader_cls=TextLoader)
        # docs = loader.load()
        
        markdown_files = list(repo_path.rglob("*.md")) + list(
            repo_path.rglob("*.mdx")
        )
        # # OR 
        # markdown_files = []
        # for dirpath, dirnames, filenames in os.walk(repo_path):
        #     for file in filenames:
        #         if file.endswith(".md") or file.endswith(".mdx"):
        #             markdown_files.append(os.path.join(dirpath, file))
        
        for markdown_file in markdown_files:
            with open(markdown_file, "r") as f:
                relative_path = markdown_file.relative_to(repo_path)
                github_url = f"https://github.com/{repo_owner}/{repo_name}/blob/{git_sha}/{relative_path}"
                yield Document(page_content=f.read(), metadata={"source": github_url})


sources = get_github_docs("yirenlu92", "deno-manual-forked")

source_chunks = []
splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
for source in sources:
    for chunk in splitter.split_text(source.page_content):
        source_chunks.append(Document(page_content=chunk, metadata=source.metadata))


# Set Up Vector DB
search_index = Chroma.from_documents(source_chunks, OpenAIEmbeddings(), persist_directory='./storage')
search_index.persist()


# Set Up LLM Chain with Custom Prompt 
from langchain.chains import LLMChain

prompt_template = """Use the context below to write a 400 word blog post about the topic below:
    Context: {context}
    Topic: {topic}
    Blog post:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "topic"])

llm = OpenAI(temperature=0)

chain = LLMChain(llm=llm, prompt=PROMPT)

# Generate text 
def generate_blog_post(topic):
    docs = search_index.similarity_search(topic, k=4)
    inputs = [{"context": doc.page_content, "topic": topic} for doc in docs]
    print(chain.apply(inputs))


generate_blog_post("environment variables")


# Using Retriever 
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=search_index.as_retriever(), return_source_documents=True)
## OR
qa = ConversationalRetrievalChain.from_llm.from_llm(ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"), search_index.as_retriever(), chain_type="map_reduce", return_source_documents=True)


query = "What did the president say about Ketanji Brown Jackson"
result = qa.run(query)
result["result"]
result["source_documents"]