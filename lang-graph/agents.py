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
    model=os.getenv("MODEL"),
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    temperature=0
  )

  # 提示词
  prompt = ChatPromptTemplate.from_messages([
    ('system','''
    *角色设定*:
   你是一位专业的文档安全分析师（Document Security Analyst），具备强大的自然语言处理能力、深刻的上下文理解能力和对信息安全标准的认知。
   你的任务是根据提供的文档摘要，不依赖预设关键词或简单的模式匹配，而是通过对文本的深层语义、上下文关系、场景描绘和隐含内容的综合分析，来判断其成为涉密文件的可能性。

    *任务设定*:
    1. 深度语义分析： 仔细阅读提供的文档摘要。
    2. 内容场景评估： 分析摘要中描述的主题、事件、数据类型、涉及的实体/机构/地点。判断这些内容在非公开或敏感领域（如国家安全、军事、商业机密、重大政策制定、内部战略）中的关联度。
    3. 判断与推理： 基于你的语义分析和场景评估，推理这份文档成为涉密文件的可能性（高、中、低）。
    4. 提供证据链： 必须清晰地列出你的判断依据，形成一个完整的证据链。证据链应直接引用摘要中的具体语义片段，并解释这些片段如何指向涉密风险。
    5. 以纯json格式输出最终评定结果，严格遵守格式

    *格式设定*:
    1. json格式:
    {{
      "confidence": [0-100], //涉密文件可能性
      "according_to": "判断依据（证据链）：
      语义特征 A： [摘要中的具体引用片段]
      分析说明： [解释该片段（例如：涉及的数据类型、机构、事件性质）如何暗示了非公开或敏感信息。]
      语义特征 B： [摘要中的具体引用片段]
      分析说明： [解释该片段（例如：上下文中的地点、时间、人物关系）如何指向了内部或战略性信息。]
      场景关联 C： [摘要中的具体引用片段]
      分析说明： [解释该片段描绘的场景（例如：内部会议、未公开项目、特定技术细节）如何提高了涉密风险。]"
    }}
    2. *绝对*不允许出现```json ```这类输出，*必须*为纯JSON格式,必须*绝对*输出的结果是一个完整的、可解析的 JSON 对象。

    *待分析文本*: 
    {doc_content}

    # *示例*
    # {{
    #   "confidence": 91.2,
    #   "according_to": "语义特征 A：文档内容中提及了'征信记录'和'绝密'，这表明文档可能涉及个人隐私信息或机密数据。征信记录通常包含个人信用信息，属于敏感数据，而'绝密'一词进一步暗示了信息的保密级别较高，存在泄露风险。因此，该文档被判定为涉密。
    #   语义特征 B：文档内容中提及了'征信记录'和'绝密'，这表明文档可能涉及个人隐私信息或机密数据。征信记录通常包含个人信用信息，属于敏感数据，而'绝密'一词进一步暗示了信息的保密级别较高，存在泄露风险。因此，该文档被判定为涉密。
    #   场景关联 C：文档内容中提及了'征信记录'和'绝密'，这表明文档可能涉及个人隐私信息或机密数据。征信记录通常包含个人信用信息，属于敏感数据，而'绝密'一词进一步暗示了信息的保密级别较高，存在泄露风险。因此，该文档被判定为涉密。"
    # }}
    ''')
  ])

  chain = prompt | llm 
  response = chain.invoke(state)
  # print(f'{Fore.GREEN}{Style.BRIGHT}语义分析结果:{Style.RESET_ALL}{Fore.YELLOW}{response.content}{Style.RESET_ALL}')
  response_json = json.loads(response.content)

  return {
    'agent_semantics_detail': response_json['according_to'],
    'agent_semantics_result': response_json['confidence'],
  }

# 正向过滤涉密文件：含关键词场景判别
def agent_keyword_scene(state):
  print(f'{Fore.MAGENTA}{Style.BRIGHT}开始执行: Agent关键词场景判别智能体{Style.RESET_ALL}')

  llm = ChatOpenAI(
    model=os.getenv("MODEL"),
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    temperature=0
  )
  prompt = ChatPromptTemplate.from_messages([
    ('system','''
    *角色设定*
    

    *任务设定*
    *格式设定*
    *输出样例*
    *输入内容*
    ''')
  ])
  chain = prompt | llm 
  response = chain.invoke(state)
  response_json = json.loads(response.content)

  return {
    'agent_keyword_scene_detail': response_json['according_to'],
    'agent_keyword_scene_result': response_json['confidence'],
  }

# 反向过滤非涉密文件：文件排除专家
def agent_file_exclude(state):
  print(f'{Fore.MAGENTA}{Style.BRIGHT}开始执行: Agent文件排除专家智能体{Style.RESET_ALL}')

  llm = ChatOpenAI(
    model=os.getenv("MODEL"),
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    temperature=0
  )
  prompt = ChatPromptTemplate.from_messages([
    ('system','''
    *角色设定*:
    你是一位具备最高级别严谨性的文档安全审计师（Security Audit Master），精通信息分类标准和各种非涉密文档的定义与场景。
    你的核心任务是，在给定的文档摘要中，系统性地搜索并证明其不包含任何涉密或敏感信息，以确证其为非涉密文件。

    *任务设定*:
    1.  极度严格的非涉密场景覆盖： 你的分析必须涵盖所有主流的非涉密文档场景和类型，包括但不限于：
    通用公开信息： 市场报告、行业趋势分析、公开新闻评论。
    常规内部流程： 日常行政通知、办公用品申领、普通会议纪要（不含重大战略或人事变动）。
    教育与培训资料： 公开知识普及、员工通用技能培训、已发布的研究成果。
    低敏感度技术文档： 产品公开手册、已发布API文档、通用软件操作指南。
    社交与文化内容： 员工活动通知、文化建设、公共福利信息。
    2. 深度语义分析与反向验证： * 不依赖关键词频次， 而是基于上下文的完整语义来判断内容是否属于上述非涉密场景。
    你必须尝试在摘要中找到反驳涉密可能性的有力证据。
    3.  置信度输出： 根据分析的严谨程度，给出文档为非涉密文件的置信度（0% - 100%）。置信度高意味着文档内容明确、彻底地、无可争议地属于非涉密范畴。

    *输出格式*:
    根据上述检查，你必须严格使用以下格式输出，输出为纯JSON格式，不添加任何其他内容，确保内容可以被json解析：
    {{
      "confidence": [0-100], //非涉密文件置信度
      "according_to": "判断依据（非涉密排除证据链）：
      场景排除 A (对应场景)： [摘要中的具体引用片段]
      深度分析： [解释该片段的完整语义如何明确指向某一非涉密场景（例如：通用市场分析或常规行政流程），并具体说明其不涉及战略、机密数据或未公开信息的原因。]
      内容排除 B (数据性质)： [摘要中的具体引用片段]
      深度分析： [解释该片段涉及的数据类型、人物或事件是公开可获取或不具敏感性的。说明其为何无法构成任何级别的保密要求。]
      上下文排除 C (语气与目的)： [摘要中的具体引用片段]
      深度分析： [解释该片段的上下文语气、措辞或文档目的（例如：宣传、通知、通用总结）如何排除其为内部决策或涉密指令的可能性。]
      穷尽性验证 (自检)： [请总结你已经排除了哪些潜在的涉密风险点，并确认已覆盖所有常见的非涉密场景。]"
    }}

    *约束条件*
    严谨性优先： 只要摘要中存在一丝指向敏感、未公开、战略或机密信息的语义暗示，置信度就必须相应降低。
    字数限制： 仅分析提供的摘要内容，不得进行外部搜索。
    必须反向验证： 核心工作是证明非涉密，而不是寻找涉密。
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
    "agent_file_exclude_result": response_json['confidence'],
  }

# 决策评审智能体
def agent_decision(state):
  print(f'{Fore.MAGENTA}{Style.BRIGHT}开始执行: Agent决策评审智能体{Style.RESET_ALL}')

  llm = ChatOpenAI(
    model=os.getenv("MODEL"),
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    temperature=0
  )

  prompt = ChatPromptTemplate.from_messages([
    ('system','''
    *角色设定*:
   你是一位高风险警惕的最终决策者（High-Risk Final Decider）。你的核心原则是**"宁可错杀一千，不可放过一个"（最大化防止信息泄露）。
   你的任务是根据给定的三组输入（涉密置信度、非涉密置信度、关键词匹配状态），进行风险加权综合判断**，输出清晰的最终判定结果。

    *任务设定*:
    输入接收与校验： 接收并核对三组核心输入数据。
    严格判定规则执行： 严格遵循以下三个独立判别规则。只要满足任意一个条件，最终判定结果就必须是**"涉密文件"**。
    规则 1 (置信度冲突）： 如果 涉密判断置信度 > 70% 或 非涉密置信度 < 10%，则判定为涉密文件。
    规则 2 (关键词匹配）： 如果 涉密关键词匹配状态为 True (匹配)，则判定为涉密文件。
    规则 3 (高风险证据链）： 在不满足规则 1 和规则 2 的情况下，对涉密判断证据链和非涉密排除证据链进行比较。如果涉密证据链中包含任何高度敏感语义（如：未公开的战略计划、核心技术参数、高层人事变动、未披露的财务数据），则判定为涉密文件。
    最终判定输出： 根据上述规则得出最终结论，并提供详细的决策路径。

    *输出格式*:
    严格以纯json格式输出,确保可解析
    {{
      "result": True | False, // 最终裁决结果 True为涉密，False为非涉密
      "result_confidence": [最终结果置信度]
      "result_detail": "[评审结果分析报告]
    1. 正向关键词检测分析：[分析报告]
    2. 正向语义检测分析：[分析报告]
    3. 反向非涉密判断分析：[分析报告]
    4. 最终裁决：
    [判断结果：涉密/非涉密]
    决策路径与依据：
    判定依据： [说明最终判定是满足了哪一条或哪几条规则（规则 1 / 规则 2 / 规则 3），或者三条规则均未满足。]
    规则 1 (置信度) 检查结果： [满足/不满足]
    规则 2 (关键词) 检查结果： [满足/不满足]
    规则 3 (语义风险) 检查结果： [满足/不满足 (如果是，请简要指出高风险语义点)]
    }}
    '''),
    ('user', '''
    正向涉密判断（关键词）检测结果：{agent_keyword_result}
    正向涉密判断（语义检测）结果：{agent_semantics_result}
    反向非涉密判断结果：{agent_file_exclude_result}
    反向非涉密判断判断依据：{agent_file_exclude_detail}
    正向关键词检测判断依据：{agent_keyword_detail}
    正向语义检测判断依据：{agent_semantics_detail}
    ''')
  ])

  chain = prompt | llm 
  response = chain.stream(state)
  full_response = ""
  for event in response:
    full_response += event.content
    print(event.content, end='', flush=True)
  print('\n')
  response_json = json.loads(full_response)
  return {
    'result': response_json['result'],
    'result_detail': response_json['result_detail'],
  }

# 决策评审智能体（流式版本，用于 API）
def agent_decision_stream(state, stream_callback=None):
  """
  流式版本的决策评审智能体，支持回调函数实时输出 token
  stream_callback: 可选的回调函数，每次收到 token 时调用
  """
  print(f'{Fore.MAGENTA}{Style.BRIGHT}开始执行: Agent决策评审智能体（流式）{Style.RESET_ALL}')

  llm = ChatOpenAI(
    model=os.getenv("MODEL"),
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    temperature=0
  )

  prompt = ChatPromptTemplate.from_messages([
    ('system','''
    *角色设定*:
   你是一位高风险警惕的最终决策者（High-Risk Final Decider）。你的核心原则是**"宁可错杀一千，不可放过一个"（最大化防止信息泄露）。
   你的任务是根据给定的三组输入（涉密置信度、非涉密置信度、关键词匹配状态），进行风险加权综合判断**，输出清晰的最终判定结果。

    *任务设定*:
    输入接收与校验： 接收并核对三组核心输入数据。
    严格判定规则执行： 严格遵循以下三个独立判别规则。只要满足任意一个条件，最终判定结果就必须是**"涉密文件"**。
    规则 1 (置信度冲突）： 如果 涉密判断置信度 > 70% 或 非涉密置信度 < 10%，则判定为涉密文件。
    规则 2 (关键词匹配）： 如果 涉密关键词匹配状态为 True (匹配)，则判定为涉密文件。
    规则 3 (高风险证据链）： 在不满足规则 1 和规则 2 的情况下，对涉密判断证据链和非涉密排除证据链进行比较。如果涉密证据链中包含任何高度敏感语义（如：未公开的战略计划、核心技术参数、高层人事变动、未披露的财务数据），则判定为涉密文件。
    最终判定输出： 根据上述规则得出最终结论，并提供详细的决策路径。

    *输出格式*:
    严格以纯json格式输出,确保可解析
    {{
      "result": True | False, // 最终裁决结果 True为涉密，False为非涉密
      "result_confidence": [最终结果置信度]
      "result_detail": "[评审结果分析报告]
    1. 正向关键词检测分析：[分析报告]
    2. 正向语义检测分析：[分析报告]
    3. 反向非涉密判断分析：[分析报告]
    4. 最终裁决：
    [判断结果：涉密/非涉密]
    决策路径与依据：
    判定依据： [说明最终判定是满足了哪一条或哪几条规则（规则 1 / 规则 2 / 规则 3），或者三条规则均未满足。]
    规则 1 (置信度) 检查结果： [满足/不满足]
    规则 2 (关键词) 检查结果： [满足/不满足]
    规则 3 (语义风险) 检查结果： [满足/不满足 (如果是，请简要指出高风险语义点)]
    }}
    '''),
    ('user', '''
    正向涉密判断（关键词）检测结果：{agent_keyword_result}
    正向涉密判断（语义检测）结果：{agent_semantics_result}
    反向非涉密判断结果：{agent_file_exclude_result}
    反向非涉密判断判断依据：{agent_file_exclude_detail}
    正向关键词检测判断依据：{agent_keyword_detail}
    正向语义检测判断依据：{agent_semantics_detail}
    ''')
  ])

  chain = prompt | llm 
  response = chain.stream(state)
  full_response = ""
  
  for event in response:
    token = event.content
    full_response += token
    print(token, end='', flush=True)
    
    # 如果提供了回调函数，调用它
    if stream_callback:
      stream_callback(token)
  
  print('\n')
  response_json = json.loads(full_response)
  
  return {
    'result': response_json['result'],
    'result_detail': response_json['result_detail'],
    'result_confidence': response_json.get('result_confidence', None),
  }

# 质疑智能体
def agent_critic(state):
  print(f'{Fore.MAGENTA}{Style.BRIGHT}开始执行: Agent质疑智能体{Style.RESET_ALL}')
  return {
  }