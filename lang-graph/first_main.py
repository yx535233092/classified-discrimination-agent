from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
import re

# 初始化 LLM
llm = ChatOpenAI(
    model="Qwen/Qwen3-32B",
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-djfnfmqslnellnmfnhmbxfktlalgdmvkshjmjpwqyenrhwwo",
    temperature=0
)

# 匹配状态 State
class MatchingState(TypedDict):
    title: str
    text: str
    keywords_to_match: List[str]
    regex_rules: List[Dict[str, str]]
    keyword_result: Dict[str, int]
    regex_result: Dict[str, List[Dict[str, Any]]]
    is_sensitive: bool
    total_keyword_matches: int
    total_regex_matches: int
    llm_analysis: str
    risk_level: str
    sensitive_topics: List[str]
    recommendations: str

# 节点1：关键词匹配
def keyword_match(state: MatchingState) -> Dict[str, Any]:
    """关键词匹配节点"""
    text = state['text'].lower()
    keywords = state['keywords_to_match']
    result = {}
    is_sensitive = False
    total_matches = 0

    for kw in keywords:
        kw_lower = kw.lower()
        matches = re.findall(re.escape(kw_lower), text)
        count = len(matches)
        if count > 0:
            result[kw] = count
            total_matches += count
            is_sensitive = True

    print(f"--- 关键词匹配完成 | 匹配 {len(result)} 个关键词 | 总计 {total_matches} 次 ---")

    return {
        'keyword_result': result,
        'is_sensitive': is_sensitive or state.get('is_sensitive', False),
        'total_keyword_matches': total_matches
    }

# 节点2：正则匹配
def regex_match(state: MatchingState) -> Dict[str, Any]:
    """正则规则匹配节点"""
    text = state['text']
    regex_rules = state['regex_rules']
    result = {}
    total_matches = 0
    
    for rule in regex_rules:
        rule_id = rule['id']
        rule_name = rule['name']
        pattern = rule['regex']
        category = rule['category']
        
        try:
            matches = re.findall(pattern, text)
            
            if matches:
                match_list = []
                for match in matches:
                    if isinstance(match, tuple):
                        matched_text = ''.join([str(m) for m in match if m])
                    else:
                        matched_text = str(match)
                    
                    match_list.append({
                        'matched_text': matched_text,
                        'category': category
                    })
                
                result[rule_id] = {
                    'rule_name': rule_name,
                    'category': category,
                    'matches': match_list,
                    'count': len(matches)
                }
                
                total_matches += len(matches)
        
        except re.error as e:
            print(f"[警告] 正则表达式错误 [{rule_id}]: {e}")
            continue
    
    is_sensitive = state.get('is_sensitive', False) or (total_matches > 0)
    
    print(f"--- 正则匹配完成 | 触发 {len(result)} 条规则 | 总计 {total_matches} 次 ---")
    
    return {
        'regex_result': result,
        'is_sensitive': is_sensitive,
        'total_regex_matches': total_matches
    }

# 节点3：LLM 深层语义检测（流式输出）
def llm_semantic_analysis(state: MatchingState) -> Dict[str, Any]:
    """LLM 深层语义分析节点 - 流式输出"""
    
    print(f"\n--- 启动 LLM 深层语义分析（流式输出）---\n")
    
    # 准备上下文信息
    text = state['text']
    title = state['title']
    keyword_matches = state['keyword_result']
    regex_matches = state['regex_result']
    total_keyword = state['total_keyword_matches']
    total_regex = state['total_regex_matches']
    
    # 构建检测摘要
    detection_summary = f"""
规则检测结果：
- 关键词匹配：{total_keyword} 次，涉及 {len(keyword_matches)} 个关键词
- 正则规则触发：{total_regex} 次，涉及 {len(regex_matches)} 条规则
- 初步判断：{'涉密' if state['is_sensitive'] else '不涉密'}
"""
    
    if keyword_matches:
        top_keywords = sorted(keyword_matches.items(), key=lambda x: x[1], reverse=True)[:10]
        keywords_str = ", ".join([f"{kw}({count}次)" for kw, count in top_keywords])
        detection_summary += f"\n主要关键词：{keywords_str}"
    
    if regex_matches:
        categories = set([details['category'] for details in regex_matches.values()])
        detection_summary += f"\n触发类别：{', '.join(categories)}"
    
    # 构建 LLM 提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位专业的信息安全分析专家，擅长识别文档中的敏感信息和潜在风险。

你的任务是：
1. 分析文档的整体内容和语境
2. 识别隐含的敏感信息（规则无法捕获的）
3. 评估信息泄露的潜在风险
4. 给出综合风险等级和处理建议

评估维度：
- 商业机密风险：是否涉及战略、财务、客户等核心商业信息
- 技术机密风险：是否涉及核心技术、源代码、算法等
- 个人隐私风险：是否涉及个人身份信息、联系方式等
- 合规风险：是否违反保密协议或法律法规
- 上下文风险：结合语境判断信息的敏感程度

风险等级定义：
- 低：一般性内部信息，泄露影响有限
- 中：部分敏感信息，需要权限控制
- 高：重要机密信息，泄露将造成重大损失
- 极高：核心机密，泄露将造成严重后果

请以 JSON 格式输出分析结果（严格遵守格式）：
{{
  "risk_level": "低/中/高/极高",
  "sensitive_topics": ["主题1", "主题2"],
  "hidden_risks": "隐含风险描述",
  "context_analysis": "上下文分析",
  "recommendations": "处理建议"
}}"""),
        ("human", """请分析以下文档的敏感信息风险：

文档标题：{title}

规则检测摘要：
{detection_summary}

文档内容（前1000字）：
{content}

请进行深层语义分析，识别隐含的敏感信息和潜在风险。""")
    ])
    
    try:
        # 流式调用 LLM
        chain = prompt | llm
        
        print("[LLM 分析中...]\n")
        
        # 收集流式输出
        full_response = ""
        for chunk in chain.stream({
            "title": title,
            "detection_summary": detection_summary,
            "content": text[:1000]
        }):
            chunk_text = chunk.content
            print(chunk_text, end="", flush=True)  # 流式打印
            full_response += chunk_text
        
        print("\n")  # 换行
        
        # 解析 LLM 返回的 JSON
        try:
            json_start = full_response.find('{')
            json_end = full_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = full_response[json_start:json_end]
                llm_result = json.loads(json_str)
            else:
                raise json.JSONDecodeError("No JSON found", "", 0)
        except json.JSONDecodeError:
            llm_result = {
                "risk_level": "中",
                "sensitive_topics": ["需人工审核"],
                "hidden_risks": "LLM 返回格式异常",
                "context_analysis": full_response,
                "recommendations": "建议人工复核"
            }
        
        # 提取结果
        risk_level = llm_result.get('risk_level', '中')
        sensitive_topics = llm_result.get('sensitive_topics', [])
        
        # 构建完整分析报告
        analysis_report = f"""
[风险等级] {risk_level}

[识别的敏感主题]
{chr(10).join(['- ' + topic for topic in sensitive_topics]) if sensitive_topics else '- 无'}

[隐含风险]
{llm_result.get('hidden_risks', '无')}

[上下文分析]
{llm_result.get('context_analysis', '无')}

[处理建议]
{llm_result.get('recommendations', '无')}
"""
        
        is_sensitive = state.get('is_sensitive', False) or (risk_level in ['高', '极高'])
        
        print(f"\n--- LLM 分析完成 | 风险等级: {risk_level} ---\n")
        
        return {
            'llm_analysis': analysis_report,
            'risk_level': risk_level,
            'sensitive_topics': sensitive_topics,
            'recommendations': llm_result.get('recommendations', ''),
            'is_sensitive': is_sensitive
        }
    
    except Exception as e:
        print(f"[错误] LLM 分析出错: {str(e)}\n")
        return {
            'llm_analysis': f"LLM 分析失败：{str(e)}",
            'risk_level': '未知',
            'sensitive_topics': [],
            'recommendations': '建议人工审核'
        }

# 节点编排
workflow = StateGraph(MatchingState)

workflow.add_node('keyword_match', keyword_match)
workflow.add_node('regex_match', regex_match)
workflow.add_node('llm_analysis', llm_semantic_analysis)

workflow.set_entry_point('keyword_match')

workflow.add_edge("keyword_match", 'regex_match')
workflow.add_edge('regex_match', 'llm_analysis')
workflow.add_edge('llm_analysis', END)

app = workflow.compile()

# ===== 使用工作流 =====

# 1. 获取测试文章内容和文章标题
with open('../秋日私语.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    title = '秋日私语'

# 2. 获取涉密关键词
with open('../rule.json', 'r', encoding='utf-8') as f:
    keywords_data = json.load(f)

# 3. 获取正则表达式规则
with open('../regex.json', 'r', encoding='utf-8') as f:
    regex_data = json.load(f)

# 解析涉密关键词
keywords_to_match = []
for category in keywords_data['categories']:
    for keyword in category['keywords']:
        keywords_to_match.append(keyword)

# 解析正则规则
regex_rules = regex_data['rules']

# 初始状态
initial_state = {
    "title": title,
    "text": text,
    "keywords_to_match": keywords_to_match,
    "regex_rules": regex_rules,
    "keyword_result": {},
    "regex_result": {},
    "is_sensitive": False,
    "total_keyword_matches": 0,
    "total_regex_matches": 0,
    "llm_analysis": "",
    "risk_level": "未知",
    "sensitive_topics": [],
    "recommendations": ""
}

print("=" * 80)
print("开始运行 LangGraph 工作流（关键词 + 正则 + LLM 三重检测）")
print("=" * 80)

# 调用 LangGraph 运行
final_state = app.invoke(initial_state)

print("\n" + "=" * 80)
print("最终检测结果")
print("=" * 80)
print(f"\n文件: {final_state['title']}")
print(f"是否涉密: {'是 [警告]' if final_state['is_sensitive'] else '否 [通过]'}")
print(f"风险等级: {final_state['risk_level']}")
print(f"关键词匹配: {final_state['total_keyword_matches']} 次")
print(f"正则匹配: {final_state['total_regex_matches']} 次")

# 显示关键词匹配详情
if final_state['keyword_result']:
    print(f"\n关键词匹配详情（TOP 10）:")
    sorted_keywords = sorted(
        final_state['keyword_result'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for kw, count in sorted_keywords[:10]:
        print(f"  - {kw}: {count} 次")
    if len(sorted_keywords) > 10:
        print(f"  ... 还有 {len(sorted_keywords) - 10} 个关键词")
else:
    print("\n[通过] 未匹配到任何关键词。")

# 显示正则匹配详情
if final_state['regex_result']:
    print(f"\n正则匹配详情（共 {len(final_state['regex_result'])} 条规则）:")
    for rule_id, details in list[Any](final_state['regex_result'].items())[:3]:
        print(f"  [{details['category']}] {details['rule_name']}: {details['count']} 次")
    if len(final_state['regex_result']) > 3:
        print(f"  ... 还有 {len(final_state['regex_result']) - 3} 条规则被触发")
else:
    print("\n[通过] 未触发任何正则规则。")

# 显示 LLM 深层分析
print(f"\nLLM 深层语义分析:")
print(final_state['llm_analysis'])

print("\n" + "=" * 80)
print("检测流程完成")
print("=" * 80)