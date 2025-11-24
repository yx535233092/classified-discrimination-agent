from typing import TypedDict
from agents import agent_keyword, agent_semantics, agent_decision, agent_critic, agent_file_exclude
from langgraph.graph import StateGraph, END
from colorama import Fore,Back,Style
import sqlite3
import os

class State(TypedDict):
  doc_title: str # 文档标题
  doc_content: str # 文档内容
  agent_keyword_result: bool # 关键词检测结果
  agent_keyword_detail: str # 关键词检测详情
  agent_semantics_result: bool # 语义检测结果
  agent_semantics_detail: str # 语义检测详情
  agent_file_exclude_result:bool 
  agent_file_exclude_detail: str 
  result: bool # 检测结果
  result_detail: str # 检测结果详情

# 开始节点
def start_node(state:State):
  print('-' * 20)
  print(f'\n{Fore.GREEN}{Style.BRIGHT}开始检测文件: {Fore.YELLOW}{Style.BRIGHT}{state["doc_title"]}{Style.RESET_ALL}\n')
  return state

# 工作流
workflow = StateGraph(State)

workflow.add_node('start_node',start_node)
workflow.add_node('agent_semantics',agent_semantics)
workflow.add_node('agent_keyword',agent_keyword)
workflow.add_node('agent_file_exclude',agent_file_exclude)
workflow.add_node('agent_decision',agent_decision)
workflow.add_node('agent_critic',agent_critic)

workflow.set_entry_point('start_node')
# 并行执行
workflow.add_edge('start_node','agent_keyword')
workflow.add_edge('start_node','agent_semantics')
workflow.add_edge('start_node','agent_file_exclude')

workflow.add_edge('agent_keyword','agent_decision')
workflow.add_edge('agent_semantics','agent_decision')
workflow.add_edge('agent_file_exclude','agent_decision')
workflow.add_edge('agent_decision',END)
# workflow.add_edge('agent_critic',END)

app = workflow.compile()

def invoke(doc_title, doc_content):
  input_state = {
    'doc_title': doc_title,
    'doc_content': doc_content,
    'agent_keyword_result': False,
    'agent_keyword_detail': '',
    'agent_semantics_result': False,
    'agent_semantics_detail': '',
  }
  final_state = app.invoke(input_state)
  return final_state

def invoke_stream(doc_title, doc_content):
  """流式执行工作流，打印过程并返回最终状态"""
  input_state = {
    'doc_title': doc_title,
    'doc_content': doc_content,
    'agent_keyword_result': False,
    'agent_keyword_detail': '',
    'agent_semantics_result': False,
    'agent_semantics_detail': '',
    'agent_file_exclude_result': False,
    'agent_file_exclude_detail': '',
  }
  
  # 收集最终状态
  final_state = {}
  
  # 遍历并打印每个节点的执行
  for event in app.stream(input_state):
    # event 是字典: {'节点名': {节点输出}}
    for node_name, node_output in event.items():
      print(f'{Fore.CYAN}节点: {node_name}{Style.RESET_ALL}')
      print(f'{Fore.YELLOW}输出: {node_output}{Style.RESET_ALL}')
      # 更新最终状态
      final_state.update(node_output)
  
  print('\n')
  return final_state

def test():
  # 读取测试数据库
  current_dir = os.path.dirname(os.path.abspath(__file__))
  conn = sqlite3.connect(os.path.join(current_dir, 'test_documents_2.db'))
  cursor = conn.cursor()
  cursor.execute('SELECT * FROM test')
  test_data = cursor.fetchall()
  conn.close()

  result_arr = []
  test_arr = []

  for data in test_data:
    input_state = {
      'doc_title': data[1],
      'doc_content': data[2],
      'agent_keyword_result': False,
      'agent_keyword_detail': '',
      'agent_semantics_result': False,
      'agent_semantics_detail': '',
      'agent_file_exclude_result': False,
      'agent_file_exclude_detail': '',
      'result': False,
      'result_detail': '',
    }

    final_state = app.invoke(input_state)
    result_arr.append(final_state['result'].__str__())
    print(f'\n{Fore.BLUE}{Style.BRIGHT}最终结果:{Style.RESET_ALL}',flush=True)
    print(f'{Fore.CYAN}{Style.BRIGHT}(正向)关键词检测结果:{Style.RESET_ALL}{Fore.YELLOW}{final_state["agent_keyword_result"]}{Style.RESET_ALL}',flush=True)
    print(f'{Fore.CYAN}{Style.BRIGHT}(正向)语义检测结果:{Style.RESET_ALL}{Fore.YELLOW}{final_state["agent_semantics_result"]}{Style.RESET_ALL}',flush=True)
    print(f'{Fore.CYAN}{Style.BRIGHT}(反向)非涉密判断结果:{Style.RESET_ALL}{Fore.YELLOW}{final_state["agent_file_exclude_result"]}{Style.RESET_ALL}',flush=True)
    print(f'{Fore.CYAN}{Style.BRIGHT}(结论)决策评审结果:{Style.RESET_ALL}{Fore.YELLOW}{final_state["result"]}{Style.RESET_ALL}',flush=True)
    print(f'{Fore.CYAN}{Style.BRIGHT}检测结果详情:{Style.RESET_ALL}{Fore.YELLOW}{final_state["result_detail"]}{Style.RESET_ALL}',flush=True)
    print('-' * 20)
    print('\n')
  print(result_arr)


  for test in test_data:
    if test[3] == 1:
      test_arr.append('True')
    else:
      test_arr.append('False')
  print(test_arr)

  successIndex = 0
  for index, test_a in enumerate(test_arr):
    if test_a == result_arr[index]:
      successIndex += 1
  print(f'{Fore.GREEN}{Style.BRIGHT}准确率:{Style.RESET_ALL}{Fore.YELLOW}{successIndex / len(test_arr) * 100}%{Style.RESET_ALL}')

# test()