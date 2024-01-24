import os 
os.environ["OPENAI_API_KEY"] = ""
# os.environ["SERPAPI_API_KEY"] = ""

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI 

from langchain.agents import create_sql_agent 
from langchain.agents import AgentExecutor 
from langchain.agents.agent_toolkits import SQLDatabaseToolkit 
from langchain.sql_database import SQLDatabase 

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ChatMessageHistory

from langchain.agents import create_pandas_dataframe_agent
# from langchain.agents.agent_types import AgentType

import pandas as pd


# llm = OpenAI(temperature=0)
# tools = load_tools(["serpapi", "llm-math"], llm=llm)

# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
# agent.run("Who is Leo DeCaprio girlfriend?")


class DatabaseQA:
    def __init__(self, db_user, db_password, db_host, db_name, db_type):
        """
        Where db_type is the database server name e.g mysql, postgresql, oracle, etc
        """
        if db_type == "mysql":
            db = SQLDatabase.from_uri(f"mysql+pymsql://{db_user}:{db_password}@{db_host}/{db_name}")
        elif db_type == "postgresql":
            db = SQLDatabase.from_uri(f"postgresql://{db_user}:{db_password}@{db_host}/{db_name}")
    
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        # chat = ChatOpenAI(model_name="gpt-4")
        
        toolkit = SQLDatabaseToolkit(db=db)
        
        return self.agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            memory=ConversationBufferMemory(),# or ChatMessageHistory()
            verbose=True
        )
    
    def run(self, query):
        return self.agent_executor.run(query)
    

def get_df(dtype, filepath):
    if dtype == "csv":
        df = pd.read_csv(filepath, sheet_name)
    elif dtype == "excel":
        df = pd.read_excel(filepath, sheet_name)
    elif dtype == "json":
        df = pd.read_json(filepath)
    return df


def get_agent(*args):
    """args = [df1, df2]"""
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
        args,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    return agent 



agent = get_agent([df1, df2])
llm_result = agent.run("how many rows in the age column are different?")
print(llm_result)
# try getting the total number of used tokens with this
print(llm_result.llm_output)
print(dir(llm_output))

