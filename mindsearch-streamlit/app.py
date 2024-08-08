import streamlit as st
import asyncio
import json
from mindsearch.agent import create_model, init_agent

# Set page title and favicon
st.set_page_config(
    page_title="MindSearch",
    page_icon="🔍"
)

# Sidebar for user input and model selection
st.sidebar.title("MindSearch")
query = st.sidebar.text_area("質問を入力してください:", "")
model_format = st.sidebar.selectbox("モデルを選択してください:", ["internlm_server", "internlm_client", "internlm_hf", "gpt4", "qwen"])
lang = st.sidebar.selectbox("言語を選択してください:", ["cn", "en"])
if st.sidebar.button("実行"):
    if query:
        with st.spinner("Thinking..."):
            # Initialize the selected model
            llm = create_model(model_format)
            agent = init_agent(llm, lang=lang)

            # Generate response using the agent
            response = ""
            references = {}
            for agent_return in agent.stream_chat(query):
                response = agent_return.response
                references = agent_return.references
                st.text_area("回答:", value=response, key="response_area")
                st.write("参考文献:")
                for key, value in references.items():
                    st.write(f"{key}: {value}")
    else:
        st.sidebar.warning("質問を入力してください")
