# %%
from model import OPENAI_MODEL
from fastapi_app.sql_operations import list_tables, describe_table, run_sql_query
from fastapi_app.dataframe import create_dataframe_pd_json

import pandas as pd
import plotly.express as px

from typing_extensions import Any, TypedDict, Optional, Annotated, Literal
from sqlalchemy import create_engine

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages

import io, re
from textwrap import dedent

# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    sql_query: Optional[str]
    dataframe: Optional[Any]
    code_generated: Optional[str]

# %%
engine = create_engine('postgresql+psycopg2://chinook:chinook@localhost:5433/chinook_auto_increment')

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
def should_continue(state: State) -> Literal["correct_query", "query_gen"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is a tool call, then we finish
    # if getattr(last_message, "tool_calls", None):
    #     return "end"
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
class CodeGenerated(BaseModel):
    """Submit the code generated to the user based on the query results."""
    code_generated: Optional[str] = Field(..., description="Plotly code generated for the user")

chartOptions = ('Scatter', 'Line', 'Histograph', 'Bargraph', 'Pie Chart')

def code_gen_node(state: State):
    # Add a node for a model to generate a query based on the question and schema
    query_gen_system_template = """
        system: You are an AI assistant specialized in creating insightful graphs and generating Python code for them.
        The data is already available in a pandas DataFrame named `df`.
        DataFrame columns: {df_columns}
        Sample of DataFrame (first 5 rows):
        {df_head}

        Your task:
        1.  Analyze the user's request and the provided DataFrame.
        2.  **Adhere to Chart Type Request:**
            a.  The user's instruction (provided as `user_input` to this agent) may specify a preferred chart type (e.g., "generate a scatter plot", "show a bar graph").
            b.  If a specific chart type is requested by the user and it is available in {chart_options} (e.g., 'Scatter', 'Line', 'Bar', etc.) and is appropriate for the data, **you MUST generate the Python code for that specific chart type.**
            c.  If the user does not specify a chart type, or if the specified type is not in {chart_options} or is clearly unsuitable for the data, then you may choose the most fitting chart type from {chart_options}. In such cases, briefly explain your choice of chart type in the 'insights'.
        3.  Generate valuable insights based on the data and the chosen chart.
        4.  Produce concise and correct Python code (using `plotly.express`) to plot the graph. The code should assume `df` (the pandas DataFrame) is pre-loaded.
        5.  The Python code should be a complete, executable script.
        6.  **Crucially, the generated Python code MUST include `fig.write_html('templates/chart.html')` to save the chart.** This allows the application to display it.
        7.  **Do NOT include `fig.show()` in the Python code**, as the chart display is handled by saving to HTML.
        8.  Return the insights and the Python code as per the `ChartResponses` model.

        If the request is unclear or cannot be fulfilled with the given data, return an `InvalidRequest` with an explanation.

        Example of Python code structure:
        ```python
        import plotly.express as px
        # df is assumed to be pre-loaded with the data

        # Your plotting code here
        # e.g., fig = px.bar(df, x='column_x', y='column_y', title='Your Chart Title')
        fig.write_html('templates/chart.html') # Save the chart as chart.html in templates folder
        ```

        Once you generate the code, invoke the CodeGenerated tool. Then fill the'code_generated' field with the Python and it must be formatted as markdown.
        """
    # Get the last tool message which contains the dataframe JSON
    tool_message = state["messages"][-1]
    df_json_content = tool_message.content
    
    # Convert the JSON string back to a pandas DataFrame
    df_from_tool = pd.read_json(io.StringIO(df_json_content))

    # Dynamically format the system prompt with the dataframe info
    formatted_query_gen_system = query_gen_system_template.format(
        df_columns=df_from_tool.columns,
        df_head=df_from_tool.head().to_markdown(),
        chart_options=chartOptions
    )

    # Create a new ChatPromptTemplate with the dynamically formatted system prompt
    code_gen_prompt = ChatPromptTemplate.from_messages(
        [("system", formatted_query_gen_system), ("placeholder", "{messages}")]
    )
    
    # Bind the tools to the model
    code_gen_model = code_gen_prompt | OPENAI_MODEL.bind_tools([CodeGenerated])
    
    # Invoke the model to generate code
    message = code_gen_model.invoke(state)
    
    # Add the AIMessage to the state's messages
    messages = state["messages"] + [message]
    code_generated = None
    for tc in message.tool_calls:
        if tc["name"] == "CodeGenerated":
            code_generated = tc["args"].get("code_generated")
        else:
            messages.append(
                ToolMessage(
                    content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call CodeGenerated to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                    tool_call_id=tc["id"],
                )
            )
    return {"messages": messages, "code_generated": code_generated, "dataframe": df_from_tool}

# %%
def execute_code_node(state: State):
    # print(state["code_generated"])
    code_blocks = re.findall(r"```python\n(.*?)```", state["code_generated"], re.DOTALL)
    full_code = "\n".join(code_blocks)
    full_code = dedent(full_code)

    # print(type(state["dataframe"]))
    exec_globals = {"df": state["dataframe"], "px": px, "pd": pd}
    try:
        exec(full_code, exec_globals)
        print("chart generated successfully")
    except Exception as e:
        print(e)
    return state
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

def should_proceed_to_code_gen(state: State) -> Literal["working_dataframe", "failed_dataframe"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is a tool call, then we finish
    # if getattr(last_message, "tool_calls", None):
    #     return END
    if last_message.content.startswith("Error:"):
        return "failed_dataframe"
    else:
        return "working_dataframe"

# Specify the edges between the nodes
# workflow.add_edge(START, "first_tool_call")
workflow.add_node("create_dataframe_node", create_dataframe_call)
workflow.add_node("code_gen", code_gen_node) # Node to generate the tool call
workflow.add_node("execute_code", execute_code_node) # Node to execute the code

workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "model_get_schema")
workflow.add_edge("model_get_schema", "describe_table_tool")
workflow.add_edge("describe_table_tool", "query_gen")
workflow.add_conditional_edges(
    "query_gen",
    should_continue,
    { "correct_query": "correct_query", "query_gen": "query_gen"}
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
workflow.add_conditional_edges(
    "create_dataframe_node",
    should_proceed_to_code_gen,
    {
        "working_dataframe": "code_gen",
        "failed_dataframe": "query_gen"
    }
)

workflow.add_edge("code_gen", "execute_code")
workflow.add_edge("execute_code", END)
workflow.set_entry_point("first_tool_call")

# Compile the workflow into a runnable
app = workflow.compile()

# %% [markdown]
# ### Visualizing the Graph

# %%
from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod

graph_png = app.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.PYPPETEER,
)

with open("graph.png", "wb") as f:
    f.write(graph_png)

display(Image(graph_png))

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
print("--Code Generated--")
print(messages["code_generated"])


# # %%
# messages["messages"][-1].tool_calls[0]["args"]

# # %%
# for event in app.stream(
#     {"messages": [("user", "Total number of artist in the db?")]}
# ):
#     print(event)

# # %%



