import json
import datetime
import httpx

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
MW_BASE_URL = 'http://localhost:8000/'
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
async def get_routes():
    """List all the routes FlySafair flies."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f'{MW_BASE_URL}public_api/flights/routes/')
        output = resp.json()
        return output['results']


@tool
async def search_available_flights(origin, destination, depart_date, number_of_adults, number_of_children, number_of_infants):
    """Search available flights, for one way tickets, given origin, destination airports in IATA codes, and departure date in iso format.
    Passegner capacity is filled using number_of_adults, number_of_children, number_of_infants which has to be a nonzero number."""
    passengers = []
    if number_of_adults:
        passengers.append({
            "passenger_type": "adult",
            "seats_required": number_of_adults
        })
    if number_of_children:
        passengers.append({
            "passenger_type": "child",
            "seats_required": number_of_children
        })
    if number_of_infants:
        passengers.append({
            "passenger_type": "infant",
            "seats_required": number_of_infants
        })
    params = {
        "origin": origin,
        "destination": destination,
        "depart_date": depart_date,
        "required_passengers": passengers
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(f'{MW_BASE_URL}public_api/pricing/fares/', json=params)
        return resp.json()


async def get_prompt():
    date_today = datetime.datetime.now().date().isoformat()
    async with httpx.AsyncClient() as client:
        resp = await client.get(f'{MW_BASE_URL}public_api/flights/routes/')
        routes = resp.json()

    departure_airports = ','.join([x['airport_code'] for x in routes['results']])
    system_prompt = f"""You are a helpful assistant. Your name is Flybot.
    You are responsible for booking and managing FlySafair bookings.
    FlySafair is a low cost domestic airline based in Southern Africa.
    Flybot is talkative and provides lots of specific details from its context.
    If you do not know the answer to a question, it truthfully says it does not know.
    You only may assist with FlySafair, bookings & flight queries. Today's date is {date_today}.
    Valid departure airports are: {departure_airports}.
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=system_prompt)),
        MessagesPlaceholder(variable_name='chat_history'),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ])
    return prompt


async def init_action(verbose=False):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)
    llm_with_tools = llm.bind_tools([
        convert_to_openai_tool(get_routes),
        convert_to_openai_tool(search_available_flights),
    ])
    tools = [
        get_routes,
        search_available_flights,
    ]
    prompt = await get_prompt()
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"],
        } | prompt | llm_with_tools | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)
    return agent_executor


if __name__ == '__main__':
    agent_executor = init_action(verbose=False)
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
        chat_history = chat_history[-5:]
        print('> ', result["output"])
