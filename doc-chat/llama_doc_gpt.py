from dotenv import load_dotenv
from llama_index import (
    ObsidianReader,
    GPTKeywordTableIndex,
    GPTVectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

import markdown
import logging
import sys 
from bs4 import BeautifulSoup

load_dotenv()
NOTES_PATH = ""
# docs = ObsidianReader(NOTES_PATH).load_data()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def read_file(file_path):
    with open(file_path, "r") as f:
        text = file.read()
        html = markdown.markdown(text) 
        
        soup = BeautifulSoup(html, "html.parser")
        
        # index all p tags 
        ps = soup.select("p").text 
        
        # filter the text output 
        ps.replace("/n", " ")
    return ps

def create_file_nodes(dir_path):
    """
    Usage Pattern:
    nodes, docs = create_file_node("../dir")
    """
    from llama_index import Document 
    from llama_index.node_parser import SimpleNodeParser 
    
    # list of Documents
    docs = [] 
    parser = SimpleNodeParser()
    
    # loop through each markdown in the dir path 
    for file_path in Path(dir_path).glob("*.md"):
        md = read_file(file_path)
        docs.append(Document(md))
    
    nodes = parser.get_nodes_from_documents(docs)
    return nodes, docs


if Path("./storage").exist():
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
else:
    nodes, docs = create_file_nodes(NOTES_PATH)
    index = GPTVectorStoreIndex.from_documents(nodes)
    index.storage_context.persist(persist_dir="./storage")
    
query_engine = index.as_query_engine()
res = query_engine.query("what is...?")