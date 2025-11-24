import { useState, useEffect, useRef } from 'react';
import './App.css';

interface StreamProgress {
  type: 'progress' | 'final' | 'stream_token';
  node?: string;
  data?: Record<string, unknown>;
  token?: string;
}

function App() {
  const [docContent, setDocContent] = useState('');
  const [docTitle, setDocTitle] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [streamingText, setStreamingText] = useState('');
  const [llmStreamText, setLlmStreamText] = useState(''); // LLM 流式输出文本

  // 用于自动滚动的 ref
  const llmStreamRef = useRef<HTMLDivElement>(null);
  const streamingTextRef = useRef<HTMLPreElement>(null);

  // 当 LLM 流式文本更新时，自动滚动到底部
  useEffect(() => {
    if (llmStreamRef.current) {
      llmStreamRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'end'
      });
    }
  }, [llmStreamText]);

  // 当检测进度文本更新时，自动滚动到底部
  useEffect(() => {
    if (streamingTextRef.current) {
      streamingTextRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'end'
      });
    }
  }, [streamingText]);

  const handleCheck = async () => {
    if (!docContent.trim()) {
      alert('请输入要检测的文本内容');
      return;
    }

    setLoading(true);
    setResult(null);
    setStreamingText('开始检测...\n');
    setLlmStreamText('');

    try {
      const res = await fetch('/api/check', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          doc_title: docTitle || '测试文档',
          doc_content: docContent
        })
      });

      if (!res.ok) {
        throw new Error('请求失败');
      }

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('无法读取响应流');
      }

      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.slice(6);
            try {
              const parsed: StreamProgress = JSON.parse(jsonStr);

              if (parsed.type === 'progress') {
                // 显示进度信息
                const nodeName = parsed.node || '未知节点';
                const nodeNameMap: Record<string, string> = {
                  start_node: '开始节点',
                  agent_keyword: '关键词检测',
                  agent_semantics: '语义检测',
                  agent_file_exclude: '非涉密判断',
                  agent_decision: '决策评审'
                };

                setStreamingText(
                  (prev) =>
                    prev + `\n✓ ${nodeNameMap[nodeName] || nodeName} 完成`
                );
              } else if (parsed.type === 'stream_token') {
                // 流式 token 输出
                if (parsed.token) {
                  setLlmStreamText((prev) => prev + parsed.token);
                }
              } else if (parsed.type === 'final') {
                // 显示最终结果
                if (parsed.data) {
                  setResult(parsed.data);
                }
                setStreamingText((prev) => prev + '\n\n✓ 检测完成！');
              }
            } catch (e) {
              console.error('解析失败:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('检测失败:', error);
      alert('检测失败，请稍后重试');
      setStreamingText('检测失败');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setDocContent('');
    setDocTitle('');
    setResult(null);
    setStreamingText('');
    setLlmStreamText('');
  };

  return (
    <div className="app-container">
      <div className="background-decoration"></div>

      <main className="main">
        <div className="header">
          <div className="icon-wrapper">
            <svg
              className="icon"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M12 2L2 7V12C2 16.97 5.69 21.5 12 23C18.31 21.5 22 16.97 22 12V7L12 2Z"
                fill="currentColor"
                opacity="0.2"
              />
              <path
                d="M12 2L2 7V12C2 16.97 5.69 21.5 12 23C18.31 21.5 22 16.97 22 12V7L12 2Z"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M9 12L11 14L15 10"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <h1>涉密文件检测系统</h1>
          <p className="subtitle">智能识别文档中的敏感信息，保护数据安全</p>
        </div>

        <div className="form-card">
          <div className="input-group">
            <label htmlFor="doc-title">文档标题（可选）</label>
            <input
              id="doc-title"
              type="text"
              className="text-input"
              placeholder="请输入文档标题"
              value={docTitle}
              onChange={(e) => setDocTitle(e.target.value)}
            />
          </div>

          <div className="input-group">
            <label htmlFor="doc-content">
              文档内容 <span className="required">*</span>
            </label>
            <textarea
              id="doc-content"
              className="textarea-input"
              rows={12}
              placeholder="请输入或粘贴需要检测的文本内容..."
              value={docContent}
              onChange={(e) => setDocContent(e.target.value)}
            ></textarea>
            <div className="char-count">{docContent.length} 字符</div>
          </div>

          <div className="button-group">
            <button
              className="btn btn-secondary"
              onClick={handleClear}
              disabled={loading}
            >
              清空
            </button>
            <button
              className="btn btn-primary"
              onClick={handleCheck}
              disabled={loading || !docContent.trim()}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  检测中...
                </>
              ) : (
                <>
                  <svg
                    className="btn-icon"
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M21 21L15 15M17 10C17 13.866 13.866 17 10 17C6.13401 17 3 13.866 3 10C3 6.13401 6.13401 3 10 3C13.866 3 17 6.13401 17 10Z"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  开始检测
                </>
              )}
            </button>
          </div>
        </div>

        {streamingText && (
          <div className="result-card">
            <h2 className="result-title">检测进度</h2>
            <pre
              ref={streamingTextRef}
              className="result-content"
              style={{ whiteSpace: 'pre-wrap' }}
            >
              {streamingText}
            </pre>
          </div>
        )}

        {llmStreamText && (
          <div className="result-card">
            <h2 className="result-title">🤖 AI 决策分析（实时生成）</h2>
            <div
              ref={llmStreamRef}
              className="result-content"
              style={{
                whiteSpace: 'pre-wrap',
                fontFamily: 'monospace',
                fontSize: '14px',
                lineHeight: '1.6',
                padding: '20px',
                backgroundColor: '#f8fafc',
                borderLeft: '4px solid #3b82f6',
                minHeight: '100px'
              }}
            >
              {llmStreamText}
              {loading && <span className="cursor-blink">▋</span>}
            </div>
          </div>
        )}

        {result && (
          <div className="result-card">
            <h2 className="result-title">最终判断结果</h2>
            <div className="result-content">
              <div
                style={{
                  padding: '20px',
                  marginBottom: '20px',
                  borderRadius: '8px',
                  backgroundColor: result.result ? '#fee2e2' : '#d1fae5',
                  border: `2px solid ${result.result ? '#ef4444' : '#10b981'}`
                }}
              >
                <h3
                  style={{
                    margin: '0 0 10px 0',
                    fontSize: '24px',
                    color: result.result ? '#991b1b' : '#065f46'
                  }}
                >
                  {result.result ? '⚠️ 涉密文件' : '✅ 非涉密文件'}
                </h3>
                {result.result_confidence !== undefined &&
                  result.result_confidence !== null && (
                    <p style={{ margin: '5px 0', fontSize: '16px' }}>
                      置信度：
                      {typeof result.result_confidence === 'number'
                        ? `${result.result_confidence as number}%`
                        : String(result.result_confidence + '%')}
                    </p>
                  )}
              </div>

              {result.result_detail !== undefined &&
                result.result_detail !== null && (
                  <div
                    style={{
                      padding: '15px',
                      backgroundColor: '#f9fafb',
                      borderRadius: '8px',
                      marginBottom: '15px'
                    }}
                  >
                    <h4 style={{ marginTop: 0 }}>详细分析：</h4>
                    <pre
                      style={{
                        whiteSpace: 'pre-wrap',
                        fontSize: '14px',
                        lineHeight: '1.6'
                      }}
                    >
                      {String(result.result_detail)}
                    </pre>
                  </div>
                )}

              <details style={{ marginTop: '15px' }}>
                <summary
                  style={{
                    cursor: 'pointer',
                    padding: '10px',
                    backgroundColor: '#f3f4f6',
                    borderRadius: '4px',
                    fontWeight: 'bold'
                  }}
                >
                  查看完整检测数据
                </summary>
                <pre
                  style={{
                    marginTop: '10px',
                    padding: '15px',
                    backgroundColor: '#f9fafb',
                    borderRadius: '8px',
                    fontSize: '12px',
                    overflow: 'auto'
                  }}
                >
                  {JSON.stringify(result, null, 2)}
                </pre>
              </details>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>© 2025 涉密文件检测系统 · 保护您的数据安全</p>
      </footer>
    </div>
  );
}

export default App;
