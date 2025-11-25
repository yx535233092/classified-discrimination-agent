from flask import Flask, request, jsonify, Response, stream_with_context
from main import invoke, invoke_stream, app as workflow_app
from agents import agent_keyword, agent_semantics, agent_non_secret_proof, agent_decision_stream
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
    
    # 发送开始消息
    yield f"data: {json.dumps({'type': 'progress', 'node': 'start_node', 'data': {}}, ensure_ascii=False)}\n\n"
    
    # 执行关键词检测
    keyword_result = agent_keyword(input_state)
    input_state.update(keyword_result)
    yield f"data: {json.dumps({'type': 'progress', 'node': 'agent_keyword', 'data': keyword_result}, ensure_ascii=False)}\n\n"
    
    # 根据关键词检测结果决定是否继续语义检测
    # 如果关键词检测结果为 True 且置信度大于 90，则跳过语义检测和非涉密证明，直接进入决策
    if keyword_result.get('agent_keyword_result') == True and keyword_result.get('agent_keyword_confidence', 0) > 90:
      # 直接进入决策评审
      pass
    else:
      # 执行语义检测
      semantics_result = agent_semantics(input_state)
      input_state.update(semantics_result)
      yield f"data: {json.dumps({'type': 'progress', 'node': 'agent_semantics', 'data': semantics_result}, ensure_ascii=False)}\n\n"
      
      # 执行非涉密证明
      proof_result = agent_non_secret_proof(input_state)
      input_state.update(proof_result)
      yield f"data: {json.dumps({'type': 'progress', 'node': 'agent_non_secret_proof', 'data': proof_result}, ensure_ascii=False)}\n\n"
    
    # 发送决策评审开始消息
    yield f"data: {json.dumps({'type': 'progress', 'node': 'agent_decision', 'data': {'status': 'started'}}, ensure_ascii=False)}\n\n"
    
    # 收集流式 token
    collected_tokens = []
    
    def stream_callback(token):
      """流式输出回调函数"""
      collected_tokens.append(token)
      # 发送流式 token
      stream_data = {
        'type': 'stream_token',
        'node': 'agent_decision',
        'token': token
      }
      # 注意：这里不能直接 yield，但我们可以在主流程中处理
    
    # 使用改造后的流式决策评审智能体
    # 由于 stream_callback 无法在生成器内部 yield，我们需要直接流式处理
    
    # 执行决策评审（流式输出）
    decision_result = {}
    full_response = ""
    
    # 检查是否满足快速判定条件
    if input_state.get('agent_keyword_result') == True and input_state.get('agent_keyword_confidence', 0) > 90:
      # 快速判定逻辑
      result_detail = '关键词检测结果为涉密，置信度为' + str(input_state['agent_keyword_confidence']) + '，最终判定为涉密'
      result_confidence = input_state['agent_keyword_confidence']
      
      # 逐字符发送结果
      for char in result_detail:
        stream_data = {
          'type': 'stream_token',
          'node': 'agent_decision',
          'token': char
        }
        yield f"data: {json.dumps(stream_data, ensure_ascii=False)}\n\n"
      
      decision_result = {
        'result': True,
        'result_detail': result_detail,
        'result_confidence': result_confidence,
        'current_node': 'END',
      }
    else:
      # 使用 LLM 流式判定
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
        你是一名高权限的信息安全决策模块。

        *输入数据*
        关键字匹配结果：{agent_keyword_result}
        关键字匹配置信度：{agent_keyword_confidence}
        关键字匹配证据：{agent_keyword_detail}
        语义推断结果：{agent_semantics_result}
        语义推断置信度：{agent_semantics_confidence}
        语义推断证据：{agent_semantics_detail}
        非涉密证明结果：{agent_non_secret_proof_result}
        非涉密证明置信度：{agent_non_secret_proof_confidence}
        非涉密证明证据：{agent_non_secret_proof_detail}

        *任务设定*:
        你的任务是接收来自三个独立分析系统（系统一：关键词匹配；系统二：深层语义推断；系统三：非涉密证明）的详细报告。
        你必须根据报告中提供的结果、置信度、证据链，结合预设的权重，执行加权平均计算和逻辑校验，最终给出关于文本是否涉密的聚合判断。
        【聚合判断权重】:
        关键词匹配分析 (M1) 权重： 40%
        深层语义分析 (M2) 权重： 30%
        非涉密证明 (M3) 权重： 30%

        *输出格式*:
        严格以纯json格式输出,确保可解析
        {{
          "result": True | False, // 最终裁决结果 True为涉密，False为非涉密
          "result_confidence": [判断最终结果置信度]
          "result_detail": [评审结果分析报告]
        1. 关键词匹配分析：[分析报告]
        2. 语义推断分析：[分析报告]
        3. 非涉密证明分析：[分析报告]
        4. 最终裁决：
        [判断结果：涉密/非涉密]
        决策路径与依据：
        判定依据： [说明最终判定是满足了哪一条或哪几条规则（规则 1 / 规则 2 / 规则 3），或者三条规则均未满足。]
        规则 1 (关键词匹配) 检查结果： [满足/不满足]
        规则 2 (语义推断) 检查结果： [满足/不满足]
        规则 3 (非涉密证明) 检查结果： [满足/不满足]
        }}
        ''')
      ])
      
      chain = prompt | llm
      response = chain.stream(input_state)
      
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
        # 清理响应内容
        content = full_response.strip()
        if content.startswith('```json'):
          content = content[7:]
        if content.startswith('```'):
          content = content[3:]
        if content.endswith('```'):
          content = content[:-3]
        content = content.strip()
        
        # 移除注释
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
          if '//' in line:
            comment_pos = line.find('//')
            cleaned_lines.append(line[:comment_pos].rstrip())
          else:
            cleaned_lines.append(line)
        content = '\n'.join(cleaned_lines)
        
        response_json = json.loads(content)
        
        # 确保 result 是布尔值
        if isinstance(response_json.get('result'), str):
          response_json['result'] = response_json['result'].lower() in ['true', 'yes', '1']
        
        decision_result = {
          'result': response_json['result'],
          'result_detail': str(response_json.get('result_detail', '')),
          'result_confidence': int(response_json.get('result_confidence', 0)),
          'current_node': 'END',
        }
      except json.JSONDecodeError as e:
        decision_result = {
          'result': False,
          'result_detail': f'解析失败: {str(e)}\n原始响应: {full_response}',
          'result_confidence': 0,
          'current_node': 'END',
        }
      except Exception as e:
        decision_result = {
          'result': False,
          'result_detail': f'处理错误: {str(e)}',
          'result_confidence': 0,
          'current_node': 'END',
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