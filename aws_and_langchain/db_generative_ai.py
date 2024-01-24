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


## READ PARAMETERS From CLOUD FORMATION 

## The stack name here should match the stack name you used when creating the cloud formation stack.
# if used a different name while creating the cloud formation stack then change this to match the name you used
CFN_STACK_NAME = "cfn-genai-mda"

stacks = boto3.client('cloudformation').list_stacks()
stack_found = CFN_STACK_NAME in [stack['StackName'] for stack in stacks['StackSummaries']]

from typing import List
def get_cfn_outputs(stackname: str) -> List:
    cfn = boto3.client('cloudformation')
    outputs = {}
    for output in cfn.describe_stacks(StackName=stackname)['Stacks'][0]['Outputs']:
        outputs[output['OutputKey']] = output['OutputValue']
    return outputs

def get_cfn_parameters(stackname: str) -> List:
    cfn = boto3.client('cloudformation')
    params = {}
    for param in cfn.describe_stacks(StackName=stackname)['Stacks'][0]['Parameters']:
        params[param['ParameterKey']] = param['ParameterValue']
    return params

if stack_found is True:
    outputs = get_cfn_outputs(CFN_STACK_NAME)
    params = get_cfn_parameters(CFN_STACK_NAME)
    glue_crawler_name = params['CFNCrawlerName']
    glue_database_name = params['CFNDatabaseName']
    glue_databucket_name = params['DataBucketName']
    region = outputs['Region']
    print(f"cfn outputs={outputs}\nparams={params}")
else:
    print("Recheck our cloudformation stack name")
    
# # RUN THE CRAWLER 
!python python_glueworkshop.py -c {glue_crawler_name}

# # STEP 1 - CONNECT TO DATABASES USING SQL ALCHEMY.

#define connections

# collect credentials from Secrets Manager
#Refer here on how to use AWS Secrets Manager - https://docs.aws.amazon.com/secretsmanager/latest/userguide/intro.html
client = boto3.client('secretsmanager')

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

#========S3=======================#
# connect to s3 using athena
## athena variables
connathena=f"athena.{region}.amazonaws.com" 
portathena='443' #Update, if port is different
schemaathena=glue_database_name #from cfn params
s3stagingathena=f's3://{glue_databucket_name}/athenaresults/'#from cfn params
wkgrpathena='primary'#Update, if workgroup is different
# tablesathena=['dataset']#[<tabe name>]
##  Create the athena connection string
connection_string = f"awsathena+rest://@{connathena}:{portathena}/{schemaathena}?s3_staging_dir={s3stagingathena}/&work_group={wkgrpathena}"
##  Create the athena  SQLAlchemy engine
engine_athena = create_engine(connection_string, echo=False)
dbathena = SQLDatabase(engine_athena)
# dbathena = SQLDatabase(engine_athena, include_tables=tablesathena)


# #============SNOWFLAKE================#
# # connect to snowflake database
# ## snowflake variables
# sf_account_id = <your snowflake account id>
# sf_secret_id =<your snowflake credentials secret id>
# dwh = <your dwh>
# db = <your database>
# schema = <your database schema>
# table = <table name>
# ## snowflake get credentials from secrets manager
# response = client.get_secret_value(SecretId=sf_secret_id)
# secrets_credentials = json.loads(response['SecretString'])
# sf_password = secrets_credentials['password']
# sf_username = secrets_credentials['username']
# ##  Create the snowflake connection string
# connection_string = f"snowflake://{sf_username}:{sf_password}@{sf_account_id}/{db}/{schema}?warehouse={dwh}"
# ##  Create the snowflake  SQLAlchemy engine
# engine_snowflake = create_engine(connection_string, echo=False)
# dbsnowflake = SQLDatabase(engine_snowflake)

# #AURORA MYSQL
# ##connect to aurora mysql
# ##aurora mysql cluster details/variables
# cluster_arn = <your cluster arn>
# secret_arn =<your cluster secret arn>
# rdsdb=<your database>
# rdsdb_tbl = [<table name>]
# ##  Create the aurora connection string
# connection_string = f"mysql+auroradataapi://:@/{rdsdb}"
# ##  Create the aurora  SQLAlchemy engine
# engine_rds = create_engine(connection_string, echo=False,connect_args=dict(aurora_cluster_arn=cluster_arn, secret_arn=secret_arn))
# dbrds = SQLDatabase(engine_rds, include_tables=rdsdb_tbl)

# #REDSHIFT
# # connect to redshift database
# ## redshift variables
# rs_secret_id = <redshift secret id>
# rs_endpoint=<redshift endpoint>
# rs_port=<redshift port>
# rs_db=<redshift database>
# rs_schema=<redshift database schema>
# ## redshift get credentials from secrets manager
# response = client.get_secret_value(SecretId=rs_secret_id)
# secrets_credentials = json.loads(response['SecretString'])
# rs_password = secrets_credentials['password']
# rs_username = secrets_credentials['username']
# ##  Create the redshift connection string
# connection_string = f"redshift+redshift_connector://{rs_username}:{rs_password}@{rs_endpoint}:{rs_port}/{rs_db}"
# engine_redshift = create_engine(connection_string, echo=False)
# dbredshift = SQLDatabase(engine_redshift)

# # Glue Data Catalog
##Provide list of all the databases where the table metadata resides after the glue successfully crawls the table
# gdc = ['redshift-sagemaker-sample-data-dev', 'snowflake','rds-aurora-mysql-employees','sagemaker_featurestore'] # mentioned a few examples here
gdc = [schemaathena] 


# # Step 2 - Generate Dynamic Prompt Templates
# # Build a consolidated view of Glue Data Catalog by combining metadata stored for all the databases in pipe delimited format.

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
"""
s3|cfn_covid_lake|cfn_covid_dataset|totaltestresults
s3|cfn_covid_lake|cfn_covid_dataset|fips
s3|cfn_covid_lake|cfn_covid_dataset|deathincrease
s3|cfn_covid_lake|cfn_covid_dataset|hospitalizedincrease
"""


# # Step 3 - Define Functions to 1/ determine the best data channel to answer the user query, 2/ Generate response to user query¶
# # In this code sample, we use the Anthropic Model to generate inferences. You can utilize SageMaker JumpStart models to achieve the same. Guidance on how to use the JumpStart Models is available in the notebook - mda_with_llm_langchain_smjumpstart_flant5xl

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

    #set the best channel from where the query can be answered
    if 'snowflake' in generated_texts: 
            channel='db'
            db=dbsnowflake 
            print("SET database to snowflake")  
    elif 'redshift'  in generated_texts: 
            channel='db'
            db=dbredshift
            print("SET database to redshift")
    elif 's3' in generated_texts: 
            channel='db'
            db=dbathena
            print("SET database to athena")
    elif 'rdsmysql' in generated_texts: 
            channel='db'
            db=dbrds
            print("SET database to rds")    
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


# # Step 4 - Run the run_query function that in turn calls the Langchain SQL Database chain to convert 'text to sql' and runs the query against the source data channel¶
# # Some samples are provided below for test runs. Uncomment the query to run.

# Enter the query
## Few queries to try out - 
#athena - Healthcare - Covid dataset
# query = """How many covid hospitalizations were reported in NY in June of 2021?"""  
query = """Which States reported the least and maximum deaths?""" 

#snowflake - Finance and Investments
# query = """Which stock performed the best and the worst in May of 2013?"""
# query = """What is the average volume stocks traded  in July of 2013?"""

#rds - Human Resources
# query = """Name all employees with birth date this month""" 
# query = """Combien d'employés sont des femmes? """ #Ask question in French - How many females are there?
# query = """How many employees were hired before 1990?"""  

#athena - Legal - SageMaker offline featurestore
# query = """How many frauds happened in the year 2023 ?"""  
# query = """How many policies were claimed this year ?""" 

#redshift - Sales & Marketing
# query = """How many tickit sales are there""" 
# query = "what was the total commision for the tickit sales in the year 2008?" 

#api - product - weather
# query = """What is the weather like right now in New York City in degrees Farenheit?"""

#Response from Langchain
response =  run_query(query)
print("----------------------------------------------------------------------")
print(f'SQL and response from user query {query}  \n  {response}')

"""
database == s3
database.table == cfn_covid_lake.cfn_covid_dataset
SET database to athena
Step complete. Channel is:  db


SQL and response from user query Which States reported the least and maximum deaths?  
   The states that reported the least deaths are American Samoa, Northern Mariana Islands and Alaska. 
The states that reported the maximum deaths are California, New York and Florida.
"""