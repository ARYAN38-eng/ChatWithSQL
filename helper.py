from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.runnables import Runnable
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
from json import JSONDecodeError
import time
load_dotenv()


def stream_data(text, delay: float = 0.08):
    for word in text.split():
        yield word + " "
        time.sleep(delay)


class chatwithsqldatabase:
    def __init__(self, db_user, db_password, db_name, db_host):
        self.db_user = db_user
        self.db_password = db_password
        self.db_name = db_name
        self.db_host = db_host
        self.db = SQLDatabase.from_uri(
            f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}")
        self.chat_model = ChatOllama(
            base_url="http://localhost:11434", model="llama3")
        self.memory = ConversationBufferWindowMemory(k=3)
        self.conversation_chain = ConversationChain(
            llm=self.chat_model,
            memory=self.memory
        )

    def message_to_sql_helper(self, query):

        response_schemas = [
            ResponseSchema(
                name="sql_query", description="The SQL query to answer the user's question.")
        ]

        parser = StructuredOutputParser.from_response_schemas(response_schemas)

        format_instructions = parser.get_format_instructions()

        prompt_1 = PromptTemplate(
            template="""================================ System Message ================================

        Given an input question, create a syntactically correct {dialect} query to run to help find the answer. You can order the results by a relevant column to return the most interesting examples in the database.

        Always query for all the columns, Unless the user specifies a specific column to query for.
        If the user asks for a specific column, you can use that column in the query. If the user asks for a specific table, you can use that table in the query. If the user asks for a specific condition, you can use that condition in the query.
        Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
        If user ask questions beyond sql questions, Just say I am sql query generator, I can only answer sql related questions.

        Only use the following tables:
        {table_info}

        {format_instructions}
        =============================== Human Message =================================
        
        Question: {input}""",
            input_variables=["input", "dialect", "table_info"],
            partial_variables={"format_instructions": format_instructions}
        )
        chain: Runnable = prompt_1 | self.chat_model | parser

        question = query
        if not question.strip().endswith("?"):
            question = question.strip() + "?"
            
        try:
            response = chain.invoke({
                "dialect": self.db.dialect,
                "table_info": self.db.get_table_info(),
                "input": question,
            })
            if "sql_query" not in response or not response["sql_query"]:
                raise ValueError(
                    "The response does not contain a valid SQL query.")
            else:
                new_query = response["sql_query"]
        except (JSONDecodeError, ValueError) as e:
            feedback_prompt = (
            f"Original Question: {question}\n"
            f"Parser Error: {str(e)}\n"
            f"Expected format: {format_instructions}"
        )
            return self.correct_sql_with_error(feedback_prompt)

        execute_query_tool = QuerySQLDataBaseTool(db=self.db)
        sql_answer = execute_query_tool.invoke(new_query)

        prompt_2 = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question  in simple language as shown in sql result.\n\n"
            f'Question: {question}\n'
            f'SQL Query: {new_query}\n'
            f'SQL Result: {sql_answer}'
        )

        return question, new_query, sql_answer, prompt_2

    def generate_final_tokens(self, prompt_2):

        return self.conversation_chain.invoke(prompt_2)

    def correct_sql_with_error(self, feedback_prompt: str):

        # Step 1: Reuse the same response schema and parser
        response_schemas = [
            ResponseSchema(
                name="sql_query",
                description="The corrected SQL query to answer the user's question."
            )
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()

        # Step 2: Create a correction prompt with format instructions
        correction_prompt = PromptTemplate(
            input_variables=["feedback"],
            partial_variables={"format_instructions": format_instructions},
            template="""
    You are an expert SQL assistant. The following SQL query produced an error.
    Given this context, correct the SQL query so it works.

    {feedback}

    {format_instructions}
    """
        )

        chain = correction_prompt | self.chat_model | parser

        try:
            result = chain.invoke({"feedback": feedback_prompt})
            corrected_sql = result["sql_query"]
        except (JSONDecodeError, ValueError) as e:
            feedback_prompt = (
            f"Original Question: {feedback_prompt}\n"
            f"Parser Error: {str(e)}\n"
            f"Expected format: {format_instructions}"
        )
            return self.correct_sql_with_error(feedback_prompt)

        try:
            execute_query_tool = QuerySQLDataBaseTool(db=self.db)
            sql_answer = execute_query_tool.invoke(corrected_sql)
        except Exception as e:
            sql_answer = f"Error: {str(e)}"
        
        prompt_2 = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question  in simple language as shown in sql result.\n\n"
            f'SQL Query: {corrected_sql}\n'
            f'SQL Result: {sql_answer}'
        )

        return "Corrected Query", corrected_sql, sql_answer, prompt_2
