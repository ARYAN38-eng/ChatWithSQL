from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOllama
from typing_extensions import TypedDict
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.runnables import Runnable
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langgraph.graph import START, StateGraph
from dotenv import load_dotenv
import os
load_dotenv()



class chatwithsqldatabase:
    def __init__(self, db_user, db_password, db_name, db_host):
        self.db_user = db_user
        self.db_password = db_password
        self.db_name = db_name
        self.db_host = db_host
        self.db = SQLDatabase.from_uri(
            f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}")

    def message_to_sql_helper(self, query):
        chat_model = ChatOllama(base_url="http://localhost:11434", model="llama3")

        response_schemas = [
            ResponseSchema(
                name="sql_query", description="The SQL query to answer the user's question.")
        ]

        parser = StructuredOutputParser.from_response_schemas(response_schemas)


        format_instructions = parser.get_format_instructions()

        prompt_1 = PromptTemplate(
            template="""================================ System Message ================================

        Given an input question, create a syntactically correct {dialect} query to run to help find the answer. You can order the results by a relevant column to return the most interesting examples in the database.

        Never query for all the columns from a specific table, only ask for the few relevant columns given the question.

        Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

        Only use the following tables:
        {table_info}

        {format_instructions}
        =============================== Human Message =================================
        
        Question: {input}""",
            input_variables=["input", "dialect", "table_info", "top_k"],
            partial_variables={"format_instructions": format_instructions}
        )
        chain: Runnable = prompt_1 | chat_model | parser
        
    
        question = query
        if not question.strip().endswith("?"):
            question = question.strip() + "?"
        response = chain.invoke({
            "dialect": self.db.dialect,
            "table_info": self.db.get_table_info(),
            "input": question,
        })
        if "sql_query" not in response or not response["sql_query"]:
            raise ValueError("The response does not contain a valid SQL query.")
        else:
            new_query= response["sql_query"]
            
            
        
        execute_query_tool=QuerySQLDataBaseTool(db=self.db)
        sql_answer=execute_query_tool.invoke(new_query)

        
        prompt_2 = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question  in simple language as shown in sql result.\n\n"
            f'Question: {question}\n'
            f'SQL Query: {new_query}\n'
            f'SQL Result: {sql_answer}'
        )
        response = chat_model.invoke(prompt_2)
        ai_answer=response.content
        
        return question, new_query,sql_answer, ai_answer 
       