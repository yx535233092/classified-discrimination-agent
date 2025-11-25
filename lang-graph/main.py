from typing import TypedDict
from agents import agent_keyword, agent_semantics, agent_decision, agent_non_secret_proof
from langgraph.graph import StateGraph, END
from colorama import Fore,Back,Style
import sqlite3
import os

class State(TypedDict):
  doc_title: str # 文档标题
  doc_content: str # 文档内容
  keywords_list: list # 关键词列表
  current_node: str # 当前节点（用于路由）
  agent_keyword_result: bool # 关键词检测结果
  agent_keyword_detail: str # 关键词检测详情
  agent_keyword_confidence: int # 关键词检测置信度
  agent_semantics_result: bool # 语义检测结果
  agent_semantics_detail: str # 语义检测详情
  agent_semantics_confidence: int # 语义检测置信度
  agent_non_secret_proof_result:bool # 非涉密证明结果
  agent_non_secret_proof_detail: str # 非涉密证明详情
  agent_non_secret_proof_confidence: int # 非涉密证明置信度
  result: bool # 检测结果
  result_detail: str # 检测结果详情
  result_confidence: int # 检测结果置信度

# 开始节点
def start_node(state:State):
  print('-' * 20)
  print(f'\n{Fore.GREEN}{Style.BRIGHT}开始检测文件: {Fore.YELLOW}{Style.BRIGHT}{state["doc_title"]}{Style.RESET_ALL}\n')
  return state

# 路由函数：根据关键词检测结果决定下一步
def route_after_keyword(state:State):
  """
  如果关键词检测到涉密内容，直接进入决策节点
  否则，继续语义检测
  """
  current_node = state.get('current_node', 'agent_semantics')
  print(f'{Fore.BLUE}路由判断: 下一个节点为 {current_node}{Style.RESET_ALL}')
  return current_node

# 工作流
workflow = StateGraph(State)

workflow.add_node('start_node',start_node)
workflow.add_node('agent_semantics',agent_semantics)
workflow.add_node('agent_keyword',agent_keyword)
workflow.add_node('agent_non_secret_proof',agent_non_secret_proof)
workflow.add_node('agent_decision',agent_decision)
# 入口
workflow.set_entry_point('start_node')

# 第一步：从开始节点到关键词检测和非涉密判断（并行）
workflow.add_edge('start_node', 'agent_keyword')
# workflow.add_edge('start_node', 'agent_non_secret_proof')

# 第二步：关键词检测后的条件路由
workflow.add_conditional_edges(
  'agent_keyword',
  route_after_keyword,
  {
    'agent_decision': 'agent_decision',  # 如果检测到关键词，直接决策
    'agent_semantics': 'agent_semantics'  # 否则继续语义检测
  }
)

workflow.add_edge('agent_semantics', 'agent_non_secret_proof')
workflow.add_edge('agent_non_secret_proof', 'agent_decision')
workflow.add_edge('agent_decision', END)

app = workflow.compile()

def invoke(doc_title, doc_content):
  input_state = {
    'doc_title': doc_title,
    'doc_content': doc_content,
    'keywords_list': [],
    'current_node': 'start_node',
    'agent_keyword_result': False,
    'agent_keyword_detail': '',
    'agent_keyword_confidence': 0,
    'agent_semantics_result': False,
    'agent_semantics_detail': '',
    'agent_semantics_confidence': 0,
    'agent_non_secret_proof_result': False,
    'agent_non_secret_proof_detail': '',
    'agent_non_secret_proof_confidence': 0,
    'result': False,
    'result_detail': '',
    'result_confidence': 0,
  }
  final_state = app.invoke(input_state)
  return final_state

def invoke_stream(doc_title, doc_content):
  """流式执行工作流，打印过程并返回最终状态"""
  input_state = {
    'doc_title': doc_title,
    'doc_content': doc_content,
    'keywords_list': [],
    'current_node': 'start_node',
    'agent_keyword_result': False,
    'agent_keyword_detail': '',
    'agent_keyword_confidence': 0,
    'agent_semantics_result': False,
    'agent_semantics_detail': '',
    'agent_non_secret_proof_result': False,
    'agent_non_secret_proof_detail': '',
    'agent_non_secret_proof_confidence': 0,
    'result': False,
    'result_detail': '',
    'result_confidence': 0,
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
  conn = sqlite3.connect(os.path.join(current_dir, 'test_documents_3.db'))
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
      'agent_keyword_confidence': 0,
      'agent_semantics_result': False,
      'agent_semantics_detail': '',
      'agent_semantics_confidence': 0,
      'agent_non_secret_proof_result': False,
      'agent_non_secret_proof_detail': '',
      'agent_non_secret_proof_confidence': 0,
      'result': False,
      'result_detail': '',
      'result_confidence': 0,
      'current_node': 'start_node'
    }

    final_state = app.invoke(input_state)
    result_arr.append(final_state['result'].__str__())
    print(f'\n{Fore.BLUE}{Style.BRIGHT}最终结果:{Style.RESET_ALL}',flush=True)
    print(f'{Fore.CYAN}{Style.BRIGHT}决策评审结果:{Style.RESET_ALL}{Fore.YELLOW}{final_state["result"]}{Style.RESET_ALL}',flush=True)
    print(f'{Fore.CYAN}{Style.BRIGHT}检测结果置信度:{Style.RESET_ALL}{Fore.YELLOW}{final_state["result_confidence"]}{Style.RESET_ALL}',flush=True)
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
# final_state = app.invoke({'doc_title':'a','doc_content':'为深入推进“放管服”改革，切实提升群众和企业的办事体验，我局拟对政务服务大厅的窗口工作效率进行全面优化。现向社会公开征求意见，重点围绕简化办事流程、优化线上预约系统、延长服务时间等具体措施。公众可通过官方网站或政务邮箱提交意见和建议。本通知旨在广泛听取民意，所有反馈意见将在汇总整理后，适时向社会公开。本次征求意见截止日期为下个月 15 日，欢迎社会各界积极参与。'})
# print(f'\n{Fore.BLUE}{Style.BRIGHT}最终结果:{Style.RESET_ALL}',flush=True)
# print(f'{Fore.CYAN}{Style.BRIGHT}决策评审结果:{Style.RESET_ALL}{Fore.YELLOW}{final_state["result"]}{Style.RESET_ALL}',flush=True)
# print(f'{Fore.CYAN}{Style.BRIGHT}检测结果置信度:{Style.RESET_ALL}{Fore.YELLOW}{final_state["result_confidence"]}{Style.RESET_ALL}',flush=True)
# print(f'{Fore.CYAN}{Style.BRIGHT}检测结果详情:{Style.RESET_ALL}{Fore.YELLOW}{final_state["result_detail"]}{Style.RESET_ALL}',flush=True)