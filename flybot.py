import os
import json
import datetime
import httpx

from typing import Any, Type

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from langchain.tools.retriever import create_retriever_tool
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
from vector_store import get_zendesk_vector_db_connection


MEMORY_KEY = "chat_history"
MW_BASE_URL = os.environ.get('MIDDLEWARE_URL')
MW_API_KEY = os.environ.get('MIDDLEWARE_API_KEY')


@tool
async def callback_request(name, phone_number):
    """
    Call back request; for when you want FlySafair to phone you back for a query.
    Requires a valid phone number, as well a name which must be obtained form the human.
    The human needs to be identified by name and phone number.
    """
    # payload = {
    #     "name": name,
    #     "msisdn": phone_number
    # }
    # headers = {
    #     'Authorization': f'Token {MW_API_KEY}',
    # }
    # async with httpx.AsyncClient() as client:
    return "Thank you {name}, we shall call you back as soon as possible on {phone_number}."


@tool
async def get_routes():
    """List all the routes FlySafair flies."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f'{MW_BASE_URL}public_api/flights/routes/')
        output = resp.json()
        return output['results']


@tool
async def get_flight_status(flight_number: str, departure_date: str):
    """
    Given a flight number and departure date, lookup flight status,
    also return departure gate and other flight times if available."""
    flight_number = ''.join([x for x in flight_number if x.isdigit()])
    async with httpx.AsyncClient() as client:
        resp = await client.get(f'{MW_BASE_URL}/public_api/flights/{flight_number}/{departure_date}/status')
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


def get_zendesk_retriever():
    db = get_zendesk_vector_db_connection()
    retriever = db.as_retriever()
    tool = create_retriever_tool(
        retriever,
        "flysafair_zendesk_articles",
        (
            "Searches and returns existing knowledge base."
            "All human question should be checked against this."
        )
    )
    return tool


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
    retrieval_tool = get_zendesk_retriever()
    llm_with_tools = llm.bind_tools([
        convert_to_openai_tool(get_routes),
        convert_to_openai_tool(search_available_flights),
        convert_to_openai_tool(get_flight_status),
        convert_to_openai_tool(retrieval_tool),
        convert_to_openai_tool(callback_request),
    ])
    tools = [
        get_routes,
        search_available_flights,
        get_flight_status,
        retrieval_tool,
        callback_request,
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
    chat_history = []
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
