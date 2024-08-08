import os
from datetime import datetime
from lagent.actions import ActionExecutor, BaseAction
from lagent.schema import ActionReturn, ActionStatusCode
from googleapiclient.discovery import build

class GoogleSearch(BaseAction):
    name = "google_search"
    description = "Perform a Google search and return the results"

    def __init__(self):
        super().__init__()
        api_key = os.environ.get("GOOGLE_API_KEY")
        search_engine_id = os.environ.get("GOOGLE_SEARCH_ENGINE_ID")
        self.service = build("customsearch", "v1", developerKey=api_key)
        self.search_engine_id = search_engine_id

    def __call__(self, query: str, num_results: int = 5) -> ActionReturn:
        try:
            result = self.service.cse().list(q=query, cx=self.search_engine_id, num=num_results).execute()
            items = result.get('items', [])
            search_results = [{'title': item['title'], 'link': item['link'], 'snippet': item['snippet']} for item in items]
            
            return ActionReturn(
                status=ActionStatusCode.SUCCESS,
                result=[{
                    "content": json.dumps(search_results, ensure_ascii=False),
                    "type": "text"
                }]
            )
        except Exception as e:
            return ActionReturn(
                status=ActionStatusCode.ERROR,
                result=[{
                    "content": f"Error during Google search: {str(e)}",
                    "type": "text"
                }]
            )

# Create an instance of GoogleSearch
google_search = GoogleSearch()

# Check if GoogleSearch is already registered before adding it
action_executor = ActionExecutor()
if "GoogleSearch" not in action_executor.actions:
    action_executor.register_action(google_search)

# The rest of your model initialization code...
import mindsearch.agent.models as llm_factory
from mindsearch.agent.mindsearch_agent import (MindSearchAgent,
                                               MindSearchProtocol)
from mindsearch.agent.mindsearch_prompt import (
    FINAL_RESPONSE_CN, FINAL_RESPONSE_EN, GRAPH_PROMPT_CN, GRAPH_PROMPT_EN,
    fewshot_example_cn, fewshot_example_en, graph_fewshot_example_cn,
    graph_fewshot_example_en, searcher_context_template_cn,
    searcher_context_template_en, searcher_input_template_cn,
    searcher_input_template_en, searcher_system_prompt_cn,
    searcher_system_prompt_en)




internlm_server = dict(type=LMDeployServer,
                       path='internlm/internlm2_5-7b-chat',
                       model_name='internlm2',
                       meta_template=INTERNLM2_META,
                       top_p=0.8,
                       top_k=1,
                       temperature=0,
                       max_new_tokens=8192,
                       repetition_penalty=1.02,
                       stop_words=['<|im_end|>'])

internlm_client = dict(type=LMDeployClient,
                       model_name='internlm2_5-7b-chat',
                       url='http://127.0.0.1:23333',
                       meta_template=INTERNLM2_META,
                       top_p=0.8,
                       top_k=1,
                       temperature=0,
                       max_new_tokens=8192,
                       repetition_penalty=1.02,
                       stop_words=['<|im_end|>'])

internlm_hf = dict(type=HFTransformerCasualLM,
                   path='internlm/internlm2_5-7b-chat',
                   meta_template=INTERNLM2_META,
                   top_p=0.8,
                   top_k=None,
                   temperature=1e-6,
                   max_new_tokens=8192,
                   repetition_penalty=1.02,
                   stop_words=['<|im_end|>'])

gpt4 = dict(type=GPTAPI,
            model_type='gpt-4-turbo',
            key=os.environ.get('OPENAI_API_KEY', 'YOUR OPENAI API KEY'))

url = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'
qwen = dict(type=GPTAPI,
            model_type='qwen-max-longcontext',
            key=os.environ.get('QWEN_API_KEY', 'YOUR QWEN API KEY'),
            openai_api_base=url,
            meta_template=[
                dict(role='system', api_role='system'),
                dict(role='user', api_role='user'),
                dict(role='assistant', api_role='assistant'),
                dict(role='environment', api_role='system')
            ],
            top_p=0.8,
            top_k=1,
            temperature=0,
            max_new_tokens=4096,
            repetition_penalty=1.02,
            stop_words=['<|im_end|>'])




def create_model(model_format):
    if model_format == 'internlm_server':
        return LMDeployServer(**internlm_server)
    elif model_format == 'internlm_client':
        return LMDeployClient(**internlm_client)
    elif model_format == 'internlm_hf':
        return HFTransformerCasualLM(**internlm_hf)
    elif model_format == 'gpt4':
        return GPTAPI(**gpt4)
    elif model_format == 'qwen':
        return GPTAPI(**qwen)
    else:
        raise ValueError(f"Unsupported model format: {model_format}")
