from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="Pro/deepseek-ai/DeepSeek-R1",
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    temperature=0,
)

# 连接数据库
db = SQLDatabase.from_uri("sqlite:///bmj.db")

# 创建 SQL Agent
agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    verbose=True
)

SYSTEM_PROMPT = '''
角色：你是一个专业的涉密文件判断助理
任务：
1. 使用sqlite数据库查询bmj表中所有数据的summary字段内容
2. 根据summary字段内容判断文件是否涉密
判断标准：
1. 如果summary字段内容涵盖了保密、涉密、机密、绝密这类信息意图表达，则认为文件涉密，否则认为文件不涉密
输出格式：json格式，返回所有文件的名称和是否涉密判断和判断依据
'''

try:  
  agent_input = f"{SYSTEM_PROMPT}"
  agent_response = agent.invoke({"input": agent_input})
  print(agent_response['output'])
except Exception as e:
  print(f"错误: {str(e)}")