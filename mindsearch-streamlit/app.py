import asyncio
import json
import logging
import streamlit as st
from sse_starlette.sse import EventSourceResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mindsearch.agent import init_agent


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
            # ... (rest of the code from your provided generate function)
                # Instead of yielding, use st.write to display the response in Streamlit
                st.write(response_json)
                # ...

        except Exception as exc:
            # ... (rest of the code from your provided generate function)

        finally:
            # ... (rest of the code from your provided generate function)

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

# ... (rest of the Streamlit app code, if needed)

# Run the Streamlit app
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8002, log_level='info')