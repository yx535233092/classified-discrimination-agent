import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from colorama import Fore,Back,Style
from dotenv import load_dotenv
import os

load_dotenv()

# 正向过滤涉密文件：关键词检测智能体
def agent_keyword(state):
  print(f'{Fore.MAGENTA}{Style.BRIGHT}开始执行: Agent关键词检测智能体{Style.RESET_ALL}')
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


# 正向过滤涉密文件：语义分析智能体
def agent_semantics(state):
  print(f'{Fore.MAGENTA}{Style.BRIGHT}开始执行: Agent语义分析智能体{Style.RESET_ALL}')

  # 大模型
  llm = ChatOpenAI(
    model="Qwen/Qwen3-14B",
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
      "is_sensitive": true,
      "according_to": "判断依据"
    }}
    2.  *绝对*不允许出现```json ```这类输出，*必须*为纯JSON格式
    3. 不要包含任何解释、说明或代码块标记（如 ```json）
    4. 必须*绝对*输出的结果是一个完整的、可解析的 JSON 对象。

    *待分析文本*: 
    {doc_content}

    *示例*
    {{
      "is_sensitive": true,
      "according_to": "文档内容中提及了'征信记录'和'绝密'，这表明文档可能涉及个人隐私信息或机密数据。征信记录通常包含个人信用信息，属于敏感数据，而'绝密'一词进一步暗示了信息的保密级别较高，存在泄露风险。因此，该文档被判定为涉密。"
    }}
    ''')
  ])

  chain = prompt | llm 
  response = chain.invoke(state)
  # print(f'{Fore.GREEN}{Style.BRIGHT}语义分析结果:{Style.RESET_ALL}{Fore.YELLOW}{response.content}{Style.RESET_ALL}')
  response_json = json.loads(response.content)

  return {
    'agent_semantics_detail': response_json['according_to'],
    'agent_semantics_result': response_json['is_sensitive'],
  }

# 反向过滤非涉密文件：文件排除专家
def agent_file_exclude(state):
  print(f'{Fore.MAGENTA}{Style.BRIGHT}开始执行: Agent文件排除专家智能体{Style.RESET_ALL}')

  llm = ChatOpenAI(
    model="Qwen/Qwen3-14B",
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    temperature=0
  )
  prompt = ChatPromptTemplate.from_messages([
    ('system','''
    *角色设定*:
    你是一个极端谨慎的“文件排除专家”。你的唯一任务是识别一份文档是否“完全确定”为非涉密文件。
    你的判断必须具有军事级别的准确性。如果存在 0.1% 的涉密可能性，你必须拒绝排除。
    你将获得文件内容，以及一系列“非涉密白名单特征”。

    *任务设定*:
    你必须逐一验证以下所有条件，才能判定该文件为非涉密：
    1.  **内容公开性检查：** 文件的核心内容是否与任何已公开的官方文件（如新闻稿、招聘信息、产品手册公开页）在语义上高度一致？
    2.  **通用性检查：** 文件的内容是否仅包含高通用性、非业务相关的词汇（如行政通知、节日祝福、通用培训资料）？
    3.  **敏感实体检查：** 文件中是否包含任何高敏感实体，例如：[客户内部代号]、[未发布技术名称]、[项目财务数据]、[未授权的内部邮箱/电话]。如果包含一个，立即拒绝排除。
    4.  **结构特征检查：** 文件的结构是否匹配公司已批准的“对外公开模板”？（例如：无内部定密标记、无特殊权限水印）。
    5.  **否定测试：** 尝试从文件中提取“可能的泄密信息”。如果你能成功提取出任何可用于商业竞争或内部决策的敏感信息片段，立即拒绝排除。

    *输出格式*:
    根据上述检查，你必须严格使用以下格式输出，输出为纯JSON格式，不添加任何其他内容，确保内容可以被json解析：
    {{
      "according_to": "
      [分析报告]
    1. 内容公开性得分 (1-100)：[分数]
    2. 敏感实体列表：[发现的实体，如果没有则填“无”]
    3. 泄密信息提取尝试：[尝试提取的结果或“提取失败”]

    【最终裁决】 (Decision)
    [选择以下一项并给出理由]
    * **排除 (EXCLUDE):** 只有当所有安全检查均通过，且你对非涉密判断的信心高于 99% 时，才选择此项。
    * **保留 (RETAIN):** 只要有一项检查未通过，或你的信心低于 99%，立即选择此项。
      ",
      "result": True | False, // 最终裁决结果 True为保留，False为排除
    }}
    '''),
    ('user', '''
    文件内容：{doc_content}
    ''')
  ])

  chain = prompt | llm 
  response = chain.invoke(state)
  response_json = json.loads(response.content)
  # print(f'{Fore.GREEN}{Style.BRIGHT}文件排除结果:{Style.RESET_ALL}{Fore.YELLOW}{response.content}{Style.RESET_ALL}')

  return {
    "agent_file_exclude_detail": response_json['according_to'],
    "agent_file_exclude_result": response_json['result'],
  }

def agent_critic(state):
  print(f'{Fore.MAGENTA}{Style.BRIGHT}开始执行: Agent质疑智能体{Style.RESET_ALL}')
  return {
  }

# 决策评审智能体
def agent_decision(state):
  print(f'{Fore.MAGENTA}{Style.BRIGHT}开始执行: Agent决策评审智能体{Style.RESET_ALL}')

  llm = ChatOpenAI(
    model="Qwen/Qwen3-14B",
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    temperature=0
  )

  prompt = ChatPromptTemplate.from_messages([
    ('system','''
    *角色设定*:
    你是一个决策评审专家，分析此前三个智能体的判断依据，评审最终结果

    *任务设定*:
    严格按照任意一个智能体判断为涉密就认为文件涉密这个规则输出判断结果，分别对三个智能体的判断依据进行分析，给出判断依据和证据链

    *输出格式*:
    严格以纯json格式输出,确保可解析
    {{
      "result": True | False,
      "result_detail": "[评审结果分析报告]
    1. 正向关键词检测分析：[分析报告]
    2. 正向语义检测分析：[分析报告]
    3. 反向文件排除分析：[分析报告]
    4. 最终裁决：[选择以下一项并给出理由]
    * **排除 (EXCLUDE):** 只有当所有安全检查均通过，且你对非涉密判断的信心高于 99% 时，才选择此项。
    * **保留 (RETAIN):** 只要有一项检查未通过，或你的信心低于 99%，立即选择此项。“
    }}
    '''),
    ('user', '''
    正向关键词检测结果：{agent_keyword_result}
    正向语义检测结果：{agent_semantics_result}
    反向文件排除结果：{agent_file_exclude_result}
    反向文件排除判断依据：{agent_file_exclude_detail}
    正向关键词检测判断依据：{agent_keyword_detail}
    正向语义检测判断依据：{agent_semantics_detail}
    ''')
  ])

  chain = prompt | llm 
  response = chain.invoke(state)
  response_json = json.loads(response.content)
  return {
    'result': response_json['result'],
    'result_detail': response_json['result_detail'],
  }

 
