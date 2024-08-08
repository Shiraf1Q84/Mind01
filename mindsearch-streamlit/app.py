import asyncio
import json
import logging
import streamlit as st
from sse_starlette.sse import EventSourceResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mindsearch import init_agent
# from mindsearch.agent import init_agent
import janus
import lagent


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='MindSearch API')
    parser.add_argument('--lang', default='cn', type=str, help='Language')
    parser.add_argument('--model_format',
                        default='internlm_server',
                        type=str,
                        help='Model format')
    return parser.parse_args()


args = parse_arguments()
app = FastAPI(docs_url='/')

app.add_middleware(CORSMiddleware,
                   allow_origins=['*'],
                   allow_credentials=True,
                   allow_methods=['*'],
                   allow_headers=['*'])


class GenerationParams(BaseModel):
    inputs: Union[str, List[Dict]]
    agent_cfg: Dict = dict()


@app.post('/solve')
async def run(request: GenerationParams):

    def convert_adjacency_to_tree(adjacency_input, root_name):

        def build_tree(node_name):
            node = {'name': node_name, 'children': []}
            if node_name in adjacency_input:
                for child in adjacency_input[node_name]:
                    child_node = build_tree(child['name'])
                    child_node['state'] = child['state']
                    child_node['id'] = child['id']
                    node['children'].append(child_node)
            return node

        return build_tree(root_name)

    async def generate():
        try:
            queue = janus.Queue()
            stop_event = asyncio.Event()

            # Wrapping a sync generator as an async generator using run_in_executor
            def sync_generator_wrapper():
                try:
                    for response in agent.stream_chat(inputs):
                        queue.sync_q.put(response)
                except Exception as e:
                    logging.exception(
                        f'Exception in sync_generator_wrapper: {e}')
                finally:
                    # Notify async_generator_wrapper that the data generation is complete.
                    queue.sync_q.put(None)

            async def async_generator_wrapper():
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, sync_generator_wrapper)
                while True:
                    response = await queue.async_q.get()
                    if response is None:  # Ensure that all elements are consumed
                        break
                    yield response
                    if not isinstance(response, tuple) and response.state == AgentStatusCode.END:
                        break
                stop_event.set()  # Inform sync_generator_wrapper to stop

            async for response in async_generator_wrapper():
                if isinstance(response, tuple):
                    agent_return, node_name = response
                else:
                    agent_return = response
                    node_name = None
                origin_adj = deepcopy(agent_return.adjacency_list)
                adjacency_list = convert_adjacency_to_tree(
                    agent_return.adjacency_list, 'root')
                assert adjacency_list[
                    'name'] == 'root' and 'children' in adjacency_list
                agent_return.adjacency_list = adjacency_list['children']
                agent_return = asdict(agent_return)
                agent_return['adj'] = origin_adj
                response_json = json.dumps(dict(response=agent_return,
                                                current_node=node_name),
                                           ensure_ascii=False)
                # Instead of yielding, use st.write to display the response in Streamlit
                st.write(response_json)

        except Exception as exc:
            msg = 'An error occurred while generating the response.'
            logging.exception(msg)
            response_json = json.dumps(
                dict(error=dict(msg=msg, details=str(exc))),
                ensure_ascii=False)
            st.write(response_json)

        finally:
            await stop_event.wait()  # Waiting for async_generator_wrapper to stop
            queue.close()
            await queue.wait_closed()

    inputs = request.inputs
    agent = init_agent(lang=args.lang, model_format=args.model_format)
    return EventSourceResponse(generate())


# Configure Streamlit
st.title("MindSearch Demo")

# Input fields for the user
inputs = st.text_area("Enter your question:")
lang = st.selectbox("Select language:", ["cn", "en"])
model_format = st.selectbox("Select model format:", ["internlm_server"])

# Button to trigger the search
if st.button("Search"):
    # Initialize the agent
    agent = init_agent(lang=lang, model_format=model_format)

    # Run the search and display the results
    with st.spinner("Searching..."):
        for response in agent.stream_chat(inputs):
            st.write(response)

# Run the Streamlit app
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8002, log_level='info')
