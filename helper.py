from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import SystemMessage
from langchain.agents import create_sql_agent
from vllm import LLM
from langchain_community.llms import Ollama


class chatwithsqldatabase:
    def __init__(self,db_user,db_password,db_name,db_host):
        self.db_user = db_user
        self.db_password = db_password
        self.db_name = db_name
        self.db_host = db_host
        self.db=SQLDatabase.from_uri(f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}")
        
    def message_to_sql_helper(self,query):
        model=Ollama(base_url="http://localhost:11434",model="llama3")
        toolkit=SQLDatabaseToolkit(db=self.db,llm=model)
        SQL_PREFIX = """You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct SQL query to run without giving query into quotes  , then look at the results of the query and return the answer.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the below tools. Only use the information returned by the below tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        To start you should ALWAYS look at the tables in the database to see what you can query.
        Do NOT skip this step.
        Then you should query the schema of the most relevant tables."""
        
        system_message = SystemMessage(content=SQL_PREFIX)
        
        agent_executor= create_sql_agent(llm=model,toolkit=toolkit,system_message=system_message,verbose=True)
        return agent_executor.run(query)