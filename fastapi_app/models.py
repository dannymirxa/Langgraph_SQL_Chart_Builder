from typing_extensions import Any, TypedDict, Optional, Annotated
from langgraph.graph.message import AnyMessage, add_messages
import os
from dotenv import load_dotenv
from typing import Optional
from typing_extensions import Annotated
from pydantic import BaseModel
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

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    sql_query: Optional[str]
    dataframe: Optional[Any]
    code_generated: Optional[str]

class Request(BaseModel):
    query: Annotated[str, MinLen(1)]

class Response(BaseModel):
    messages: list[AnyMessage]
    sql_query: Optional[str] = None
    dataframe: Optional[Any] = None
    code_generated: Optional[str] = None
