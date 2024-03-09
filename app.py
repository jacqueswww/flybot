import json
import requests

from typing import Any, Type

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.prompts import (
    PromptTemplate,
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.prompts.chat import (
    MessagesPlaceholder,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import tool
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)

store = {}
MEMORY_KEY = "chat_history"
chat_history = []


class MultiplySchema(BaseModel):
    """Multiply tool schema."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class Multiply(BaseTool):
    args_schema: Type[BaseModel] = MultiplySchema
    name: str = "multiply"
    description: str = "Multiply two integers together."

    def _run(self, a: int, b: int, **kwargs: Any) -> Any:
        return a * b


@tool
def get_routes():
    """List all the routes FlySafair flies."""
    return requests.get('https://fa-api-prod.bluemarket.io//public_api/flights/routes/').json()


system_prompt = """You are a helpful assistant. Your name is Flybot. You are responsible for booking and managing FlySafair bookings.
FlySafair is a low cost domestic airline based in Southern Africa.
Flybot is talkative and provides lots of specific details from its context.
If you do not know the answer to a question, it truthfully says it does not know.
You only may assist with FlySafair, bookings & flight queries.
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=system_prompt)),
    MessagesPlaceholder(variable_name='chat_history'),
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
    MessagesPlaceholder(variable_name='agent_scratchpad')
])


def init_action():
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)
    llm_with_tools = llm.bind_tools([
        convert_to_openai_tool(Multiply()),
        convert_to_openai_tool(get_routes)
    ])
    tools = [
        Multiply(),
        get_routes,
    ]
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"],
        } | prompt | llm_with_tools | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor


agent_executor = init_action()
while True:
    inp = input()
    if inp.lower() == 'exit':
        break
    result = agent_executor.invoke({"input": inp, "chat_history": chat_history})
    chat_history.extend(
        [
            HumanMessage(content=inp),
            AIMessage(content=result["output"]),
        ]
    )
