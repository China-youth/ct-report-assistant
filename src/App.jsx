import { useState } from 'react'

const NODE_LABELS = {
  vision: '影像理解',
  retrieval: '知识检索',
  generate: '生成报告',
  check: '一致性自检',
}

export default function App() {
  const [ctImage, setCtImage] = useState(null)
  const [ctDesc, setCtDesc] = useState('')
  const [report, setReport] = useState(null)
  const [nodeTimings, setNodeTimings] = useState([])
  const [selfcheck, setSelfcheck] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [iteration, setIteration] = useState(0)
  const [disclaimer, setDisclaimer] = useState('')
  const [refuseReason, setRefuseReason] = useState('')

  // conversation id persisted in sessionStorage so history / stop work
  function getConvId() {
    let id = sessionStorage.getItem('ct-report-conversation-id')
    if (!id) {
      id = 'ct-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
      sessionStorage.setItem('ct-report-conversation-id', id)
    }
    return id
  }

  async function onFile(e) {
    const file = e.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => setCtImage(ev.target.result)
    reader.readAsDataURL(file)
  }

  async function generate() {
    if (!ctDesc.trim() && !ctImage) {
      setError('请填写影像所见描述，或至少上传一张 CT 图。')
      return
    }
    setLoading(true)
    setError(null)
    setReport(null)
    setRefuseReason('')
    setSelfcheck(null)
    setNodeTimings([])
    setIteration(0)

    // build the payload — send only what the user gave us
    const payload = {
      description: ctDesc.trim() || undefined,
      image: ctImage || undefined, // data URL, optional
    }

    const convId = getConvId()
    try {
      const res = await fetch('/report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'makers-conversation-id': convId,
        },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        const t = await res.text()
        throw new Error('HTTP ' + res.status + ': ' + t.slice(0, 400))
      }
      const data = await res.json()
      // backend returns { report, refuse_reason, image_status, selfcheck, node_timings, iteration, disclaimer }
      setReport(data.report || null)
      setRefuseReason(data.refuse_reason || '')
      setSelfcheck(data.selfcheck || null)
      setNodeTimings(data.node_timings || [])
      setIteration(data.iteration ?? 0)
      setDisclaimer(data.disclaimer || '')
    } catch (err) {
      setError(err.message || String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="wrap">
      <aside className="sidebar">
        <div className="brand">脑 CT 报告助手</div>
        <div className="sub">LangGraph · 影像理解 → 检索 → 生成 → 自检</div>
        <ul className="nodes">
          {Object.entries(NODE_LABELS).map(([k, label]) => {
            const t = nodeTimings.find((n) => n.node === k)
            return (
              <li key={k} className={report && t ? 'done' : ''}>
                <span className="dot" />
                <span>{label}</span>
                {t && <span className="ms">{t.ms} ms</span>}
              </li>
            )
          })}
        </ul>
      </aside>

      <main>
        <section className="card">
          <h1>影像输入</h1>
          <label className="drop" htmlFor="ctimg">
            {ctImage ? (
              <img src={ctImage} alt="ct" className="thumb" />
            ) : (
              <span className="placeholder">点击上传脑 CT 图（VLM 自动读图生成所见；也可下方手填）</span>
            )}
            <input id="ctimg" type="file" accept="image/*" onChange={onFile} hidden />
          </label>
          <textarea
            value={ctDesc}
            onChange={(e) => setCtDesc(e.target.value)}
            placeholder="（可选）手填或粘贴放射科「影像所见」描述；上传了 CT 图时此项作为补充锚点，例如：双侧大脑半球对称，脑室系统无扩大，未见明显出血及梗死灶。"
            rows={4}
          />
          <button className="cta" disabled={loading} onClick={generate}>
            {loading ? '正在生成报告…' : '生成报告'}
          </button>
        </section>

        {error && <div className="err">出错了：{error}</div>}

        {refuseReason && (
          <section className="card refuse">
            <h1>未能生成报告</h1>
            <p className="refuse-msg">{refuseReason}</p>
            {disclaimer && <p className="disclaimer">{disclaimer}</p>}
          </section>
        )}

        {report && (
          <section className="card">
            <h1>
              结构化报告 <span className="iter">迭代轮次 {iteration}</span>
            </h1>
            <ReportView report={report} />
            {selfcheck && <SelfcheckView sc={selfcheck} />}
            {disclaimer && <p className="disclaimer">{disclaimer}</p>}
          </section>
        )}
      </main>
    </div>
  )
}

function ReportView({ report }) {
  return (
    <div className="report">
      <div className="field">
        <h2>所见（所見 / Findings）</h2>
        <p>{report.findings || report.所见 || '—'}</p>
      </div>
      <div className="field">
        <h2>印象（印象 / Impression）</h2>
        <ul>
          {(report.impression || report.印象 || []).map((x, i) => (
            <li key={i}>{x}</li>
          ))}
          {(!report.impression && !report.印象) && <li>—</li>}
        </ul>
      </div>
      <div className="field">
        <h2>建议（建议 / Recommendation）</h2>
        <ul>
          {(report.recommendation || report.recommendations || report.建议 || []).map((x, i) => (
            <li key={i}>{x}</li>
          ))}
        </ul>
      </div>
    </div>
  )
}

function SelfcheckView({ sc }) {
  const issues = sc.issues || []
  const ok = sc.passed === true
  return (
    <div className="selfcheck">
      <div className="sc-title">一致性自检 {ok ? '通过' : '未通过'}</div>
      {issues.length > 0 ? (
        <ul>
          {issues.map((x, i) => (
            <li key={i}>{x}</li>
          ))}
        </ul>
      ) : (
        <p>未发现幻觉、超范围诊断或术语错误。</p>
      )}
    </div>
  )
}
