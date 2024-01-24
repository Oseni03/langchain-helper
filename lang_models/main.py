import pandas as pd

# Download and save the dataset in a Pandas dataframe.
DATASET_URL='https://github.com/GoogleCloudPlatform/python-docs-samples/raw/main/cloud-sql/postgres/pgvector/data/retail_toy_dataset.csv'
df = pd.read_csv(DATASET_URL)
df = df.loc[:, ['product_id', 'product_name', 'description', 'list_price']]
# df.head(5)


# Save the Pandas dataframe in a PostgreSQL table.
import asyncio
import asyncpg
from google.cloud.sql.connector import Connector

async def main():
    loop = asyncio.get_running_loop()
    async with Connector(loop=loop) as connector:
        # Create connection to Cloud SQL database
        conn: asyncpg.Connection = await connector.connect_async(
            f"{project_id}:{region}:{instance_name}",  # Cloud SQL instance connection name
            "asyncpg",
            user=f"{database_user}",
            password=f"{database_password}",
            db=f"{database_name}"
        )

        # Create the `products` table.   
        await conn.execute("""CREATE TABLE products(
                                product_id VARCHAR(1024) PRIMARY KEY,
                                product_name TEXT,
                                description TEXT,
                                list_price NUMERIC)""")

        # Copy the dataframe to the `products` table.
        tuples = list(df.itertuples(index=False))
        await conn.copy_records_to_table('products', records=tuples, columns=list(df), timeout=10)
        await conn.close()

# Run the SQL commands now.
await main() # type: ignore


#####  Generating the vector embeddings using Vertex AI #####
# Split long text into smaller chunks with LangChain

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    separators = [".", "\n"],
    chunk_size = 500,
    chunk_overlap  = 0,
    length_function = len,
)
chunked = []
for index, row in df.iterrows():
  product_id = row['product_id']
  desc = row['description']
  splits = text_splitter.create_documents([desc])
  for s in splits:
    r = { 'product_id': product_id, 'content': s.page_content }
    chunked.append(r)


"""
After you split long product descriptions 
into smaller chunks, you can generate 
vector embeddings for each chunk by using 
the Text Embedding Model available through 
Vertex AI. 
"""

from langchain.embeddings import VertexAIEmbeddings
from google.cloud import aiplatform

aiplatform.init(project=f"{project_id}", location=f"{region}")
embeddings_service = VertexAIEmbeddings()

batch_size = 5
for i in range(0, len(chunked), batch_size):
  request = [x['content'] for x in chunked[i: i + batch_size]]
  response = embeddings_service.embed_documents(request)
  # Post-process the generated embeddings.
  # ...


"""
Use pgvector to store the generate embeddings

After creating the pgvector extension and 
registering a new vector data type, you can 
store a NumPy array directly into a 
PostgreSQL table.
"""

from pgvector.asyncpg import register_vector

# ...
await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
await register_vector(conn)


# Create the `product_embeddings` table to store vector embeddings.
await conn.execute("""
    CREATE TABLE product_embeddings(
        product_id VARCHAR(1024), 
        content TEXT,
        embedding vector(768))
""")


# Store all the generated embeddings.
for index, row in product_embeddings.iterrows():
    await conn.execute(
        "INSERT INTO product_embeddings VALUES ($1, $2, $3)"
        row['product_id'], 
        row['content'], 
        np.array(row['embedding'])
    )


###### Finding similar toys using pgvector cosine search operator ######

"""
Step 1: Generate the vector embedding for the incoming input query.
"""
# Generate vector embedding for the user query.
from langchain.embeddings import VertexAIEmbeddings
embeddings_service = VertexAIEmbeddings()
qe = embeddings_service.embed_query([user_query])

"""
Step 2: Use the new pgvector cosine similarity search operator to find related products 

Notice how you can combine the vector search operation with the regular SQL filters on the `list_price` column using the powerful PostgreSQL and pgvector query semantics.
"""

# Use cosine similarity search to find the top five products 
# that are most closely related to the input query.

results = await conn.fetch("""
    WITH vector_matches AS (
        SELECT product_id, 
               1 - (embedding <=> $1) AS similarity
        FROM product_embeddings
        WHERE 1 - (embedding <=> $1) > $2
        ORDER BY similarityDESC
        LIMIT $3
    )
    SELECT 
        product_name, 
        list_price, 
        description 
    FROM products
    WHERE product_id IN (SELECT product_id FROM vector_matches)
        AND list_price >= $4 AND list_price <= $5
""", qe, similarity_threshold, num_matches, min_price, max_price)



# Using LangChain for summarization and efficient context building.

from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain
from IPython.display import display, Markdown

llm = VertexAI()

map_prompt_template = """
You will be given a detailed description of a toy product.
This description is enclosed in triple backticks (```).
Using this description only, extract the name of the toy,
the price of the toy and its features. 
```{text}```
SUMMARY:
"""
map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

combine_prompt_template = """
You will be given a detailed description different toy products
enclosed in triple backticks (```) and a question enclosed in
double backticks(``).
Select one toy that is most relevant to answer the question.
Using that selected toy description, answer the following
question in as much detail as possible.
You should only use the information in the description.
Your answer should include the name of the toy, the price of
the toy and its features. 
Your answer should be less than 200 words.
Your answer should be in Markdown in a numbered list format.

Description:
```{text}```
Question:
``{user_query}``
Answer:
"""

combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text", "user_query"])

docs = [Document(page_content=t) for t in matches]
chain = load_summarize_chain(llm,
    chain_type="map_reduce",
    map_prompt=map_prompt,
    combine_prompt=combine_prompt)
answer = chain.run({
    'input_documents': docs,
    'user_query': user_query,
})


display(Markdown(answer))