# %%
from models import OPENAI_MODEL, InvalidRequest, SQLResponse, ChartResponses
from sql_operations import list_tables, describe_table, run_sql_query
from pydantic import Field
from annotated_types import MinLen

import pandas as pd
from sqlalchemy import Engine, create_engine
from sqlalchemy.inspection import inspect
from typing_extensions import TypedDict, Optional, Annotated, Sequence
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langchain_core.runnables.config import RunnableConfig
import operator

engine = create_engine('postgresql+psycopg2://chinook:chinook@localhost:5433/chinook_auto_increment')

# %%
class AgentState(TypedDict):
    db_engine: Engine = engine
    question: str
    tables: list[str] = None
    query: str
    query_result: str
    dataframe: pd.DataFrame
    error_message: str

# %%
graph = StateGraph(AgentState)

# %%
state= AgentState(question="Show me how many albums each artist has, and plot this as a bar chart. List the artists and their album counts.", db_engine=engine)

# %%
def list_tables_tool(state: AgentState) -> list[str]:
    state["tables"] = eval(list_tables(state['db_engine']))
    return state

# %%
list_tables_tool(state)

# %%
def describe_table_tool(state: AgentState, tables: list[str]) -> str:
    schema = [describe_table(state["db_engine"], table) for table in tables]
    return schema

# %%
class SQLSuccess(BaseModel):
    sql_query: Annotated[str, MinLen(1)]
    detail: str = Field(alias='Detail', description='Explanation of the SQL query, steps taken, the result of the query (JSON), and chart generation summary if applicable, as markdown')
    chart_insights: Optional[Annotated[str, MinLen(1)]] = Field(None, description="Insights from the dataset and graph, if a chart was generated.")
    chart_python_code: Optional[Annotated[str, MinLen(1)]] = Field(None, description='Python code to plot the graph, as markdown, if a chart was generated.')


# %%
class tablesNames(BaseModel):
    tables: Annotated[list[str], Field(description="list of table names relevant to the question")]

# %%
def create_nl_to_sql(state: AgentState):
    question = state["question"]
    tables = state["tables"]
    
    system = f"""You are an assistant that list out relevant tables that are relevant to the user's messages:

    List of tables:
        {tables}

    Understand user's message carefully and pick only the tables available in the list
    """

    human = f"Question: {question}"
    check_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )

    llm = OPENAI_MODEL
    structured_llm = llm.with_structured_output(tablesNames)
    relevance_checker = check_prompt | structured_llm
    relevance = relevance_checker.invoke({})
    print(f"Table Names: {state['tables']}")
    return state

# %%
create_nl_to_sql(state)

# %%



