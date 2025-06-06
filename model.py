import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import TypeAlias, Union, Optional
from typing_extensions import Annotated
from pydantic import Field
from annotated_types import MinLen


load_dotenv('.env')

from langchain_openai import AzureChatOpenAI

OPENAI_MODEL = AzureChatOpenAI(
    azure_deployment="gpt-4o-dev",  # or your deployment
    api_version="2024-08-01-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint="https://llmcoechangemateopenai2.openai.azure.com/",
)

class SQLSuccess(BaseModel):
    sql_query: Annotated[str, MinLen(1)]
    detail: str = Field(alias='Detail', description='Explanation of the SQL query, steps taken, the result of the query (JSON), and chart generation summary if applicable, as markdown')
    chart_insights: Optional[Annotated[str, MinLen(1)]] = Field(None, description="Insights from the dataset and graph, if a chart was generated.")
    chart_python_code: Optional[Annotated[str, MinLen(1)]] = Field(None, description='Python code to plot the graph, as markdown, if a chart was generated.')

class InvalidRequest(BaseModel):
    error_message: str

SQLResponse: TypeAlias = Union[SQLSuccess, InvalidRequest]

class ChartResponses(BaseModel):
    insights: Annotated[str, MinLen(1), Field(alias='insights', description="insights from the dataset and graph")]
    python_code: Annotated[str, MinLen(1), Field(alias='python_code',description='Python code to plot the graph, as markdown')]
