import json
import boto3

import sqlalchemy
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL

from langchain.docstore.document import Document
from langchain import PromptTemplate,SagemakerEndpoint,SQLDatabase, SQLDatabaseChain, LLMChain
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import SQLDatabaseSequentialChain

from langchain.chains.api.prompt import API_RESPONSE_PROMPT
from langchain.chains import APIChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatAnthropic
from langchain.chains.api import open_meteo_docs

from typing import Dict
import time 


# ===========provide user input===========#
glue_databucket_name = 'blog-genai-mda' #Create this bucket in S3
glue_db_name='genai-workshop200'
glue_role=  'AWSGlueServiceRole-glueworkshop200'
glue_crawler_name=glue_db_name+'-crawler200'


## Create IAM role that run the crawler 

import boto3
import os
# Retrieve the AWS account number
sts_client = boto3.client('sts')
account_number = sts_client.get_caller_identity().get('Account')
# Retrieve the AWS region
#region = os.environ['AWS_REGION']
region = boto3.session.Session().region_name
print("AWS Account Number:", account_number)
print("AWS Region:", region)
trust_policy="""{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "",
      "Effect": "Allow",
      "Principal": {
        "Service": "glue.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}"""
managed_policy="""{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Action": [
                "glue:*"
            ],
            "Resource": [
                "arn:aws:glue:"""+region+""":"""+account_number+""":catalog",
                "arn:aws:glue:"""+region+""":"""+account_number+""":database/*",
                "arn:aws:glue:"""+region+""":"""+account_number+""":table/*"
            ],
            "Effect": "Allow",
            "Sid": "Readcrawlerresources"
        },
        {
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": [
                "arn:aws:glue:"""+region+""":"""+account_number+""":log-group:/aws-glue/crawlers*",
                "arn:aws:logs:*:*:/aws-glue/*",
                "arn:aws:logs:*:*:/customlogs/*"
            ],
            "Effect": "Allow",
            "Sid": "ReadlogResources"
        },
    {
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:PutBucketLogging",
                "s3:ListBucket",
                "s3:PutBucketVersioning"
            ],
            "Resource": [
                "arn:aws:s3:::"""+glue_databucket_name+"""",
                "arn:aws:s3:::"""+glue_databucket_name+"""/*"
            ],
            "Effect": "Allow",
            "Sid": "ReadS3Resources"
        }
    ]
    }"""
print(managed_policy, file=open('managed-policy.json', 'w'))
print(trust_policy, file=open('trust-policy.json', 'w'))


""" terminal 
%%sh -s "$glue_role" 
echo $1 
glue_role="$1"
managed_policy_name="managed-policy-$glue_role"
echo $managed_policy_name
aws iam create-role --role-name $glue_role --assume-role-policy-document file://trust-policy.json
output=$(aws iam create-policy --policy-document file://managed-policy.json --policy-name $managed_policy_name)
arn=$(echo "$output" | grep -oP '"Arn": "\K[^"]+')
echo "$arn"
aws iam attach-role-policy --policy-arn $arn --role-name $glue_role
"""


import boto3

client = boto3.client('glue')

# Create database 
try:
    response = client.create_database(
        DatabaseInput={
            'Name': glue_db_name,
            'Description': 'This database is created using Python boto3',
        }
    )
    print("Successfully created database")
except:
    print("error in creating database. Check if the database already exists")

#introducing some lag for the iam role to create
time.sleep(20) 

# Create Glue Crawler 
try:

    response = client.create_crawler(
        Name=glue_crawler_name,
        Role=glue_role,
        DatabaseName=glue_db_name,
        Targets={
            'S3Targets': [
                {
                    'Path': 's3://{BUCKET_NAME}/covid-dataset/'.format(BUCKET_NAME =glue_databucket_name)
                }
            ]
        },
        TablePrefix=''
    )
    
    print("Successfully created crawler")
except:
    print("error in creating crawler. However, if the crawler already exists, the crawler will run.")

# Run the Crawler
try:
    response = client.start_crawler(Name=glue_crawler_name )
    print("Successfully started crawler. The crawler may take 2-5 mins to detect the schema.")
    while True:
        # Get the crawler status
        response = client.get_crawler(Name=glue_crawler_name)
         # Extract the crawler state
        status = response['Crawler']['State']
        # Print the crawler status
        print(f"Crawler '{glue_crawler_name}' status: {status}")
        if status == 'READY':  # Replace 'READY' with the desired completed state
            break  # Exit the loop if the desired state is reached

        time.sleep(10)  # Sleep for 10 seconds before checking the status again
    
except:
    print("error in starting crawler. Check the logs for the error details.")


# # Step 1 - Connect to databases using SQL Alchemy.

#define connections

# collect credentials from Secrets Manager
#Refer here on how to use AWS Secrets Manager - https://docs.aws.amazon.com/secretsmanager/latest/userguide/intro.html
client = boto3.client('secretsmanager')
region=client.meta.region_name

#LLM 
#get the llm api key
#llm variables
#Refer here for access to Anthropic API Keys https://console.anthropic.com/docs/access
anthropic_secret_id = "anthropic"#<your anthropic secret id>
## llm get credentials from secrets manager
response = client.get_secret_value(SecretId=anthropic_secret_id)
secrets_credentials = json.loads(response['SecretString'])
ANTHROPIC_API_KEY = secrets_credentials['ANTHROPIC_API_KEY']
#define large language model here. Make sure to set api keys for the variable ANTHROPIC_API_KEY
llm = ChatAnthropic(temperature=0, anthropic_api_key=ANTHROPIC_API_KEY, max_tokens_to_sample = 512)

#S3
# connect to s3 using athena
## athena variables
connathena=f"athena.{region}.amazonaws.com" 
portathena='443' #Update, if port is different
schemaathena=glue_db_name #from user defined params
s3stagingathena=f's3://{glue_databucket_name}/athenaresults/'#from cfn params
wkgrpathena='primary'#Update, if workgroup is different
# tablesathena=['dataset']#[<tabe name>]
##  Create the athena connection string
connection_string = f"awsathena+rest://@{connathena}:{portathena}/{schemaathena}?s3_staging_dir={s3stagingathena}/&work_group={wkgrpathena}"
##  Create the athena  SQLAlchemy engine
engine_athena = create_engine(connection_string, echo=False)
dbathena = SQLDatabase(engine_athena)

gdc = [schemaathena] 


# # Step 2 - Generate Dynamic Prompt Templates

#Generate Dynamic prompts to populate the Glue Data Catalog
#harvest aws crawler metadata

def parse_catalog():
    #Connect to Glue catalog
    #get metadata of redshift serverless tables
    columns_str=''
    
    #define glue cient
    glue_client = boto3.client('glue')
    
    for db in gdc:
        response = glue_client.get_tables(DatabaseName =db)
        for tables in response['TableList']:
            #classification in the response for s3 and other databases is different. Set classification based on the response location
            if tables['StorageDescriptor']['Location'].startswith('s3'):  classification='s3' 
            else:  classification = tables['Parameters']['classification']
            for columns in tables['StorageDescriptor']['Columns']:
                    dbname,tblname,colname=tables['DatabaseName'],tables['Name'],columns['Name']
                    columns_str=columns_str+f'\n{classification}|{dbname}|{tblname}|{colname}'                     
    #API
    ## Append the metadata of the API to the unified glue data catalog
    columns_str=columns_str+'\n'+('api|meteo|weather|weather')
    return columns_str

glue_catalog = parse_catalog()

#display a few lines from the catalog
print('\n'.join(glue_catalog.splitlines()[-10:]) )


# # Step 3 - Define Functions to 1/ determine the best data channel to answer the user query, 2/ Generate response to user query

#Function 1 'Infer Channel'
#define a function that infers the channel/database/table and sets the database for querying
def identify_channel(query):
    #Prompt 1 'Infer Channel'
    ##set prompt template. It instructs the llm on how to evaluate and respond to the llm. It is referred to as dynamic since glue data catalog is first getting generated and appended to the prompt.
    prompt_template = """
     From the table below, find the database (in column database) which will contain the data (in corresponding column_names) to answer the question 
     {query} \n
     """+glue_catalog +""" 
     Give your answer as database == 
     Also,give your answer as database.table == 
     """
    ##define prompt 1
    PROMPT_channel = PromptTemplate( template=prompt_template, input_variables=["query"]  )

    # define llm chain
    llm_chain = LLMChain(prompt=PROMPT_channel, llm=llm)
    #run the query and save to generated texts
    generated_texts = llm_chain.run(query)
    print(generated_texts)

    #set the channel from where the query can be answered
    if 's3' in generated_texts: 
            channel='db'
            db=dbathena
            print("SET database to athena")
    elif 'api' in generated_texts: 
            channel='api'
            print("SET database to weather api")        
    else: raise Exception("User question cannot be answered by any of the channels mentioned in the catalog")
    print("Step complete. Channel is: ", channel)
    
    return channel, db

#Function 2 'Run Query'
#define a function that infers the channel/database/table and sets the database for querying
def run_query(query):

    channel, db = identify_channel(query) #call the identify channel function first

    ##Prompt 2 'Run Query'
    #after determining the data channel, run the Langchain SQL Database chain to convert 'text to sql' and run the query against the source data channel. 
    #provide rules for running the SQL queries in default template--> table info.

    _DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.

    Do not append 'Query:' to SQLQuery.
    
    Display SQLResult after the query is run in plain english that users can understand. 

    Provide answer in simple english statement.
 
    Only use the following tables:

    {table_info}
    If someone asks for the sales, they really mean the tickit.sales table.
    If someone asks for the sales date, they really mean the column tickit.sales.saletime.

    Question: {input}"""

    PROMPT_sql = PromptTemplate(
        input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
    )

    
    if channel=='db':
        db_chain = SQLDatabaseChain.from_llm(llm, db, prompt=PROMPT_sql, verbose=True, return_intermediate_steps=False)
        response=db_chain.run(query)
    elif channel=='api':
        chain_api = APIChain.from_llm_and_api_docs(llm, open_meteo_docs.OPEN_METEO_DOCS, verbose=True)
        response=chain_api.run(query)
    else: raise Exception("Unlisted channel. Check your unified catalog")
    return response


# # Step 4 - Run the run_query function that in turn calls the Langchain SQL Database chain to convert 'text to sql' and runs the query against the source data channel

# Enter the query
## Few queries to try out - 
#athena - Healthcare - Covid dataset
# query = """How many covid hospitalizations were reported in NY in June of 2021?"""  
query = """Which States reported the least and maximum deaths?""" 

#api - product - weather
# query = """What is the weather like right now in New York City in degrees Farenheit?"""

#Response from Langchain
response =  run_query(query)
print("----------------------------------------------------------------------")
print(f'SQL and response from user query {query}  \n  {response}')

"""
database == s3|genai-workshop200|covid_dataset
database.table == s3|genai-workshop200|covid_dataset
SET database to athena
Step complete. Channel is:  db

SQL and response from user query Which States reported the least and maximum deaths?  
  The states that reported the least deaths are American Samoa, Northern Mariana Islands and Alaska.
The states that reported the maximum deaths are California, New York and Florida.
"""