import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from colorama import Fore,Back,Style
from dotenv import load_dotenv
import os

load_dotenv()

# 关键词检测智能体
def agent_keyword(state):
  print(f'{Fore.GREEN}{Style.BRIGHT}1.关键词检测智能体开始执行{Style.RESET_ALL}')
  # 加载关键词库
  keywords_list = []
  with open('lang-graph/lib/keywords.json', 'r', encoding='utf-8') as f:
    keywords = json.load(f)
    for category in keywords:
        for keyword in keywords[category]:
          keywords_list.append(keyword)

  # 正则匹配
  for keyword in keywords_list:
    matches = re.findall(rf'{keyword}', state['doc_content'])
    if matches:
      if state['agent_keyword_result'] == False:
        state['agent_keyword_result'] = True
      state['agent_keyword_detail'] += f'{keyword}: {len(matches)}, \n'

  # print(f'\n{Fore.GREEN}{Style.BRIGHT}关键词检测结果:{Style.RESET_ALL}\n{Fore.YELLOW}{Style.BRIGHT}{state["agent_keyword_detail"]}{Style.RESET_ALL}')
  return {
    'agent_keyword_result': state['agent_keyword_result'],
    'agent_keyword_detail': state['agent_keyword_detail'],
  }


# 语义分析智能体
def agent_semantics(state):
  print(f'{Fore.GREEN}{Style.BRIGHT}2.语义分析智能体开始执行{Style.RESET_ALL}')

  # 大模型
  llm = ChatOpenAI(
    model="Qwen/Qwen3-32B",
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    temperature=0
  )

  # 提示词
  prompt = ChatPromptTemplate.from_messages([
    ('system','''
    *角色设定*:
    你是一名资深的信息安全审计专家，精通国家保密法规、企业安全策略和金融/技术领域的敏感信息分类。
    你的任务是判断以下文本是否包含任何层级的涉密或敏感信息。你必须关注语义推理，而非仅仅关键词匹配。

    *任务设定*:
    1. 分析文档的整体内容和语境
    2. 给出判断依据
    3. 以纯json格式输出最终评定结果，以True表示涉密，False表示非涉密，严格遵守格式

    *场景设定*:

    *格式设定*:
    1. json格式:
    {{
      "is_sensitive": True,
      "according_to": "判断依据"
    }}
    2.  *绝对*不允许出现```json ```这类输出，*必须*为纯JSON格式

    *待分析文本*: 
    {doc_content}

    *示例*
    {{
      "is_sensitive": True,
      "according_to": "文档内容中提及了'征信记录'和'绝密'，这表明文档可能涉及个人隐私信息或机密数据。征信记录通常包含个人信用信息，属于敏感数据，而'绝密'一词进一步暗示了信息的保密级别较高，存在泄露风险。因此，该文档被判定为涉密。"
    }}
    ''')
  ])

  chain = prompt | llm 
  response = chain.invoke(state)
  print(f'{Fore.GREEN}{Style.BRIGHT}语义分析结果:{Style.RESET_ALL}{Fore.YELLOW}{response.content}{Style.RESET_ALL}')
  response_json = json.loads(response.content)
  print(response_json)

  return {
    'agent_semantics_detail': response_json['according_to'],
    'agent_semantics_result': response_json['is_sensitive'],
  }

# 决策评审智能体
def agent_decision(state):
  print(f'{Fore.GREEN}{Style.BRIGHT}3.决策评审智能体开始执行{Style.RESET_ALL}')

  return {
    'result': state['agent_keyword_result'] or state['agent_semantics_result'],
    'result_detail': state['agent_keyword_detail'] + state['agent_semantics_detail'],
  }