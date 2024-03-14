import json
import uuid
import traceback
import sys

from sanic import Sanic
from sanic import Request, Websocket

from flybot import init_action
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

app = Sanic("FlybotApp")


@app.websocket("/feed")
async def feed(request: Request, ws: Websocket):
    chat_history = [
        AIMessage(
            content=("Hello there! My name is FlyBot,"
                     "I am here to assist you with FlySafair bookings and general queries.")
        ),
    ]
    print('init_action')
    agent_executor = await init_action()
    print(agent_executor)
    while True:
        data = await ws.recv()
        try:
            msg = json.loads(data)
            inp = msg['text'][:255]  # safety shunt.
            print('text: ', inp)
            msg_id = str(uuid.uuid4())
            output_str = ''

            # result = await agent_executor.ainvoke({
            #     "input": inp,
            #     "chat_history": chat_history
            # })
            # print(result)
            # output_str = result['output']
            # msg_payload = {
            #     'id': msg_id,
            #     'text': output_str,
            # }

            async for s in agent_executor.astream({
                    "input": inp,
                    "chat_history": chat_history}):
                print('->>>' + str(s), end="", flush=True)
                if s.get('actions'):
                    msg_payload = {
                        'id': msg_id,
                        'type': 'action',
                        'text': '',
                    }
                if s.get('output'):
                    msg_payload = {
                        'id': msg_id,
                        'type': 'output',
                        'text': s['output'],
                    }
                    output_str += s['output']
                await ws.send(json.dumps(msg_payload))

            chat_history.extend(
                [
                    HumanMessage(content=inp),
                    AIMessage(content=output_str),
                ]
            )
            chat_history = chat_history[-5:]
            await ws.send(json.dumps(msg_payload))
        except Exception as e:
            print(traceback.format_exc())
            print(sys.exc_info()[2])
            await ws.close()

        # print(ws)
        # data = "hello!"
        # print("Sending: " + data)
        # await ws.send(data)
        # data = await ws.recv()
        # print("Received: " + data)


app.static("/", "./frontend/", index='index.html')

if __name__ == "__main__":
    app.run()
