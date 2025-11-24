from flask import Flask, request, jsonify, Response, stream_with_context
from main import invoke, invoke_stream, app as workflow_app
from agents import agent_keyword, agent_semantics, agent_file_exclude, agent_decision_stream
import json

app = Flask(__name__)

@app.route('/check', methods=['POST'])
def check():
  data = request.json
  doc_title = data.get('doc_title')
  doc_content = data.get('doc_content')
  
  def generate():
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
    
    # 发送开始消息
    yield f"data: {json.dumps({'type': 'progress', 'node': 'start_node', 'data': {}}, ensure_ascii=False)}\n\n"
    
    # 执行关键词检测
    keyword_result = agent_keyword(input_state)
    input_state.update(keyword_result)
    yield f"data: {json.dumps({'type': 'progress', 'node': 'agent_keyword', 'data': keyword_result}, ensure_ascii=False)}\n\n"
    
    # 执行语义检测
    semantics_result = agent_semantics(input_state)
    input_state.update(semantics_result)
    yield f"data: {json.dumps({'type': 'progress', 'node': 'agent_semantics', 'data': semantics_result}, ensure_ascii=False)}\n\n"
    
    # 执行非涉密判断
    exclude_result = agent_file_exclude(input_state)
    input_state.update(exclude_result)
    yield f"data: {json.dumps({'type': 'progress', 'node': 'agent_file_exclude', 'data': exclude_result}, ensure_ascii=False)}\n\n"
    
    # 发送决策评审开始消息
    yield f"data: {json.dumps({'type': 'progress', 'node': 'agent_decision', 'data': {'status': 'started'}}, ensure_ascii=False)}\n\n"
    
    # 执行决策评审（流式输出）
    decision_result = {}
    
    def stream_callback(token):
      """流式输出回调函数"""
      stream_data = {
        'type': 'stream_token',
        'node': 'agent_decision',
        'token': token
      }
      # 注意：这里不能直接 yield，需要通过 nonlocal 变量传递
      pass
    
    # 使用流式版本的决策评审
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    import os
    
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
    response = chain.stream(input_state)
    full_response = ""
    
    # 逐个 token 发送
    for event in response:
      token = event.content
      full_response += token
      
      # 发送流式 token
      stream_data = {
        'type': 'stream_token',
        'node': 'agent_decision',
        'token': token
      }
      yield f"data: {json.dumps(stream_data, ensure_ascii=False)}\n\n"
    
    # 解析最终结果
    try:
      response_json = json.loads(full_response)
      decision_result = {
        'result': response_json['result'],
        'result_detail': response_json['result_detail'],
        'result_confidence': response_json.get('result_confidence', None),
      }
    except json.JSONDecodeError as e:
      decision_result = {
        'result': False,
        'result_detail': f'解析失败: {str(e)}\n原始响应: {full_response}',
        'result_confidence': 0,
      }
    
    input_state.update(decision_result)
    
    # 发送决策评审完成消息
    yield f"data: {json.dumps({'type': 'progress', 'node': 'agent_decision', 'data': decision_result}, ensure_ascii=False)}\n\n"
    
    # 发送最终结果
    final_data = {
      'type': 'final',
      'data': input_state
    }
    yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
  
  return Response(stream_with_context(generate()), 
                  mimetype='text/event-stream',
                  headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                  })

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5001, debug=True)