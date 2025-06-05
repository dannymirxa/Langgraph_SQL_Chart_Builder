# %%
from models import OPENAI_MODEL
from sql_operations import list_tables, describe_table, run_sql_query
from dataframe import create_dataframe_pd_json, create_dataframe_pd
import pandas as pd

from sqlalchemy import create_engine
from langchain_community.utilities.sql_database import SQLDatabase

from typing_extensions import Any, TypedDict, Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph, END
import re
from textwrap import dedent
import uuid, io
import plotly.express as px


# %%
engine = create_engine('postgresql+psycopg2://chinook:chinook@localhost:5433/chinook_auto_increment')

# %%
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

def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


# %%
from langchain_core.tools import tool

# %%
from typing import Annotated, Literal

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages

# %%
from langchain_core.prompts import ChatPromptTemplate

# %%
# create_dataframe_pd_json_tool("SELECT * FROM artist;")

# %%
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    code_generated: Optional[str]
    dataframe: Optional[Any]

# %%
workflow = StateGraph(State)

# %% [markdown]
# ### Create dataframe

# %%
@tool
def create_dataframe_pd_json_tool(query: str = None) -> str:
    """This tool receives the SQL query and return pandas dataframe from the result"""
    return create_dataframe_pd_json(engine, query)

# %%
query =  """SELECT g.Name AS Genre, SUM(il.unit_price * il.Quantity) AS TotalSales
                                FROM invoice_line il
                                JOIN Track t ON il.track_id = t.track_id
                                JOIN Genre g ON t.genre_id = g.genre_id
                                GROUP BY g.Name
                                ORDER BY TotalSales DESC;"""

# %%

# %%
def create_dataframe_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_dataframe_pd_json_tool",
                        "args": { "query": query },
                        "id": str(uuid.uuid4()),
                    }
                ],
            )
        ],
    }

# %% [markdown]
# ### Code Generation

# %%
class CodeGenerated(BaseModel):
    """Submit the code generated to the user based on the query results."""
    code_generated: Optional[str] = Field(..., description="Plotly code generated for the user")

# %%
chartOptions = ('Scatter', 'Line', 'Histograph', 'Bargraph', 'Pie Chart')

# %%
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

# %%
# state = State(messages=["Build a bar chart to show the Total Sales by Genre"])
# message = code_gen.invoke(state)

# %%
def code_gen_node(state: State):
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
# Define a conditional edge to decide whether to continue or end the workflow
def should_continue(state: State) -> Literal["correct_code", "code_gen"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is a tool call, then we finish
    # if getattr(last_message, "tool_calls", None):
    #     return END
    if last_message.content.startswith("Error:"):
        return "code_gen"
    else:
        return "correct_code"

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
workflow.add_node("generate_dataframe_tool_call", create_dataframe_call)
workflow.add_node("execute_dataframe_tool", create_tool_node_with_fallback([create_dataframe_pd_json_tool]))
workflow.add_node("code_gen", code_gen_node)
workflow.add_node("execute_code", execute_code_node)
workflow.add_conditional_edges(
    "code_gen",
    should_continue,
    {
        "correct_code": "execute_code",
        "code_gen": "code_gen"
    }
)

workflow.add_edge("generate_dataframe_tool_call", "execute_dataframe_tool")
workflow.add_edge("execute_dataframe_tool", "code_gen")
workflow.add_edge("execute_code", END)
workflow.set_entry_point("generate_dataframe_tool_call")


# %%
app = workflow.compile()

# %%
from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod

# %%
display(
    Image(
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)

# %%
messages = app.invoke(
    {"messages": [("user", "Build a bar chart to show the Total Sales by Genre")]}
)

# %%
messages["messages"][-1].content

# %%
code_blocks = re.findall(r"```python\n(.*?)```", messages["messages"][-1].content, re.DOTALL)
full_code = "\n".join(code_blocks)
full_code = dedent(full_code)
print(full_code)

# %%



