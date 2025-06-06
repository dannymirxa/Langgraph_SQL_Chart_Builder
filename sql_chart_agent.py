# %%
from models import OPENAI_MODEL
from sql_operations import list_tables, describe_table, run_sql_query
from dataframe import create_dataframe_pd_json

import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities.sql_database import SQLDatabase

from typing_extensions import Any, TypedDict, Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph, END

from typing import Annotated, Literal

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
import uuid, io


# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    sql_query: Optional[str]
    dataframe: Optional[Any]
    code_generated: Optional[str]


# %%
engine = create_engine('postgresql+psycopg2://chinook:chinook@localhost:5433/chinook_auto_increment')

db = SQLDatabase(engine)

# %% [markdown]
# ### Creating Utility Functions
# 
# ##### 1. It allows the creation of tool nodes with built-in error handling.
# ##### 2. If a tool execution fails, instead of crashing, it captures the error.
# ##### 3. It then formats this error into a message that the agent can understand and act upon.
# ##### 4. This allows the agent to attempt to correct its mistakes or try alternative approaches when a tool fails, making the overall system more resilient and capable of handling unexpected situations.

# %%
def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

# %% [markdown]
# ### Defining Tools for the Agent

# %%
from langchain_core.tools import tool
# %%

# Do not pass SQLAlchemyEngine here because it cannot be serialized as JSON schema
@tool
def list_tables_tool() -> str:
    """This tool lists all tables in the db in string representation of a list"""
    return list_tables(engine)

@tool
def describe_table_tool(table_name: str = None) -> str:
    """This tool receives a table name and describes the table in the db as a string"""
    return describe_table(engine, table_name)

@tool
def run_sql_query_tool(query: str = None):
    """This tool receives an sql query, runs it in the engine, and returns the result as a string"""
    return run_sql_query(engine, query)

# %%
from langchain_core.prompts import ChatPromptTemplate

# %% 
## Create query checker function
def _create_query_checker():
    query_check_system = """You are a SQL expert with a strong attention to detail.
                Double check the Postgres query for common mistakes, including:
                - Using NOT IN with NULL values
                - Using UNION when UNION ALL should have been used
                - Using BETWEEN for exclusive ranges
                - Data type mismatch in predicates
                - Properly quoting identifiers
                - Using the correct number of arguments for functions
                - Casting to the correct data type
                - Using the proper columns for joins

                If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

                You will call the appropriate tool to execute the query after running this check."""

    query_check_prompt = ChatPromptTemplate.from_messages(
        [("system", query_check_system), ("placeholder", "{messages}")]
        )
    query_check = query_check_prompt.pipe(OPENAI_MODEL.bind_tools(
        [run_sql_query_tool], tool_choice="required"
        ))

    return query_check

# %%
def model_check_query(state: State) -> dict[str, Any]:
    """
    Use this tool to double-check if your query is correct before executing it.
    """
    query_check = _create_query_checker()
    message = query_check.invoke({"messages": [state["messages"][-1]]})
    sql_query = None
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] == "run_sql_query_tool" and "query" in tc["args"]:
                sql_query = tc["args"]["query"]
                break
    return {"messages": [message], "sql_query": sql_query}

# %%
# Add a node for the first tool call
# retun messages because it is an object in the state
def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_tables_tool",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }

# %%
# Define a new graph
workflow = StateGraph(State)

# %%
workflow.add_node("first_tool_call", first_tool_call)

# Add nodes for the first two tools
workflow.add_node("list_tables_tool", create_tool_node_with_fallback([list_tables_tool]))
workflow.add_node("describe_table_tool", create_tool_node_with_fallback([describe_table_tool]))

# Add a node for a model to choose the relevant tables based on the question and available tables
model_get_schema = OPENAI_MODEL.bind_tools([describe_table_tool])
workflow.add_node(
                "model_get_schema",
                lambda state: {
                    "messages": [model_get_schema.invoke(state["messages"])],
                },
                )

# %%
# Describe a tool to represent the end state
class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""
    final_answer: str = Field(..., description="The final answer to the user")
    sql_query: Optional[str] = Field(None, description="The SQL query that was executed to get the answer")

# %% 
# Add a node for a model to generate a query based on the question and schema

def _create_query_generator():    
    query_gen_system = """You are a SQL expert with a strong attention to detail.

        Given an input question, output a syntactically correct Postgres query to run. After the query is executed and you have the results, formulate a human-readable answer based on those results.

        DO NOT call any tool besides SubmitFinalAnswer to submit the final answer.

        When generating the query:

        Output the SQL query that answers the input question without a tool call.

        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.

        If you get an error while executing a query, rewrite the query and try again.

        If you get an empty result set, you should try to rewrite the query to get a non-empty result set.
        NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.

        Once you have the query results, use them to construct a clear, concise, and human-readable answer to the original question. Then, invoke the SubmitFinalAnswer tool. The 'final_answer' argument of this tool should be the human-readable answer you constructed. The 'sql_query' argument should be the SQL query that was executed.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""
    query_gen_prompt = ChatPromptTemplate.from_messages(
            [("system", query_gen_system), ("placeholder", "{messages}")]
        )
    query_gen = query_gen_prompt.pipe(OPENAI_MODEL.bind_tools(
            [SubmitFinalAnswer]
        ))
    return query_gen

# %%
def query_gen_node(state: State):
    query_gen = _create_query_generator()
    message = query_gen.invoke(state)

    # Sometimes, the LLM will hallucinate and call the wrong tool. We need to catch this and return an error message.
    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] == "SubmitFinalAnswer":
                # Add the stored SQL query to the arguments of SubmitFinalAnswer
                if state.get("sql_query"):
                    tc["args"]["sql_query"] = state["sql_query"]
            else:
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                        tool_call_id=tc["id"],
                    )
                )
    else:
        tool_messages = []
    return {"messages": [message] + tool_messages}

# %%

workflow.add_node("query_gen", query_gen_node)

# Add a node for the model to check the query before executing it
workflow.add_node("correct_query", model_check_query)

# Add node for executing the query
workflow.add_node("execute_query", create_tool_node_with_fallback([run_sql_query_tool]))

# %%
# Define a conditional edge to decide whether to continue or end the workflow
def should_continue(state: State) -> Literal["end", "correct_query", "query_gen"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is a tool call, then we finish
    if getattr(last_message, "tool_calls", None):
        return "end"
    if last_message.content.startswith("Error:"):
        return "query_gen"
    else:
        return "correct_query"
    
# %%
    
def create_dataframe_call(state: State) -> dict[str, Any]:
    sql_query = state.get("sql_query")
    if sql_query:
        try:
            dataframe_json_str = create_dataframe_pd_json(engine, sql_query)
            df = pd.read_json(io.StringIO(dataframe_json_str))
            return {"dataframe": df}
        except Exception as e:
            return {
                "messages": [
                    AIMessage(
                        content=f"Error creating dataframe: {repr(e)}\n Please fix the SQL query or data issue.",
                        tool_calls=[],
                    )
                ]
            }
    else:
        return {"messages": [AIMessage(content="Error: SQL query not found in state to create dataframe.")]}


# %%

# Define a new conditional edge to decide whether to generate dataframe or continue query gen
def should_proceed_to_dataframe(state: State) -> Literal["generate_dataframe", "continue_query_gen"]:
    # This is a simplified logic. In a real application, you'd have a more robust way
    # to determine if a dataframe is needed (e.g., based on user's initial intent).
    # For this task, we assume if a SQL query was successfully executed, we proceed to dataframe generation.
    messages = state["messages"]
    last_message = messages[-1]
    # Check if the last message is a ToolMessage and if sql_query is present in the state
    if state.get("sql_query") and isinstance(last_message, ToolMessage) and last_message.tool_call_id:
        return "generate_dataframe"
    return "continue_query_gen"

# Specify the edges between the nodes
# workflow.add_edge(START, "first_tool_call")
workflow.add_node("create_dataframe_node", create_dataframe_call) # Node to generate the tool call
workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "model_get_schema")
workflow.add_edge("model_get_schema", "describe_table_tool")
workflow.add_edge("describe_table_tool", "query_gen")
workflow.add_conditional_edges(
    "query_gen",
    should_continue,
    { "correct_query": "correct_query", "query_gen": "query_gen", "end": END}
)
workflow.add_edge("correct_query", "execute_query")
workflow.add_conditional_edges( # Modify the edge from execute_query
    "execute_query",
    should_proceed_to_dataframe,
    {
        "generate_dataframe": "create_dataframe_node", # Transition to the node that generates the tool call
        "continue_query_gen": "query_gen"
    }
)
workflow.add_edge("create_dataframe_node", END) # After executing the dataframe tool, end for now
workflow.set_entry_point("first_tool_call")

# Compile the workflow into a runnable
app = workflow.compile()

# %% [markdown]
# ### Visualizing the Graph

# %%
from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod

display(
    Image(
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)

# %% [markdown]
# ### Running the Agent

# %%
messages = app.invoke(
    {"messages": [("user", "Total number of albums for each artist with pop genre")]}
)

# %%
print("--Latest Message--")
print(messages["messages"][-1].content)
print("--SQL Query--")
print(messages["sql_query"])
print("--Latest Message--")
print(messages["dataframe"])

# %%
messages["messages"][-1].tool_calls[0]["args"]

# %%
for event in app.stream(
    {"messages": [("user", "Total number of artist in the db?")]}
):
    print(event)

# %%



