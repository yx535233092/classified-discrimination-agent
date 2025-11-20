from typing import TypedDict
from agents import agent_keyword, agent_semantics, agent_decision
from langgraph.graph import StateGraph, END
from colorama import Fore,Back,Style

class State(TypedDict):
  doc_title: str # 文档标题
  doc_content: str # 文档内容
  agent_keyword_result: bool # 关键词检测结果
  agent_keyword_detail: str # 关键词检测详情
  agent_semantics_result: bool # 语义检测结果
  agent_semantics_detail: str # 语义检测详情
  result: bool # 检测结果
  result_detail: str # 检测结果详情

def start_node(state:State):

  print(f'{Fore.GREEN}{Style.BRIGHT}开始检测文件: {Fore.YELLOW}{Style.BRIGHT}{state["doc_title"]}{Style.RESET_ALL}')
  return state

workflow = StateGraph(State)

workflow.add_node('start_node',start_node)
workflow.add_node('agent_keyword',agent_keyword)
workflow.add_node('agent_semantics',agent_semantics)
workflow.add_node('agent_decision',agent_decision)

workflow.set_entry_point('start_node')
workflow.add_edge('start_node','agent_keyword')
workflow.add_edge('start_node','agent_semantics')
workflow.add_edge('agent_keyword','agent_decision')
workflow.add_edge('agent_semantics','agent_decision')
workflow.add_edge('agent_decision',END)

app = workflow.compile()

# 输入文本
input_state = {
  'doc_title': '测试',
  'doc_content': '征信记录绝密的绝密的', # 文档内容
  'agent_keyword_result': False,
  'agent_keyword_detail': '',
  'agent_semantics_result': False,
  'agent_semantics_detail': '',
  'result': False,
  'result_detail': '',
}

final_state = app.invoke(input_state)

print(f'\n{Fore.GREEN}最终结果:{Style.RESET_ALL}{Fore.YELLOW}{final_state}{Style.RESET_ALL}')


