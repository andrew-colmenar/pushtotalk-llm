import React, { useEffect, useRef, useState } from 'react'

type Turn = { role: 'user' | 'assistant', text: string }

export default function App() {
  const [activeTab, setActiveTab] = useState<'chat'|'llm'>('chat')
  const [turns, setTurns] = useState<Turn[]>([])
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState(false)
  const [payloadText, setPayloadText] = useState('')
  const [payloadJson, setPayloadJson] = useState<object|null>(null)
  const scrollerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetch('/history').then(r => r.json()).then(data => {
      setTurns(data.turns || [])
    }).catch(() => {})
  }, [])

  useEffect(() => {
    const el = scrollerRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [turns])

  async function send(e: React.FormEvent) {
    e.preventDefault()
    const text = message.trim()
    if (!text || busy) return
    setTurns(t => [...t, { role: 'user', text }])
    setMessage('')
    setBusy(true)
    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
      })
      const data = await res.json()
      if (res.ok) {
        setTurns(t => [...t, { role: 'assistant', text: data.answer || '' }])
        setPayloadText(data.payload_text || '')
        setPayloadJson(data.payload || null)
      } else {
        setTurns(t => [...t, { role: 'assistant', text: 'Error: ' + (data.error || res.status) }])
      }
    } catch (err) {
      setTurns(t => [...t, { role: 'assistant', text: 'Network error' }])
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{display:'flex',flexDirection:'column',height:'100vh',background:'#0b1120',color:'#e5e7eb'}}>
      <div style={{padding:'12px 16px',background:'#111827',borderBottom:'1px solid #1f2937',fontWeight:600}}>PTT Chat UI (Vite)</div>

      <div style={{display:'flex',gap:8,padding:'8px 12px',background:'#0b1325',borderBottom:'1px solid #1f2937'}}>
        <button onClick={()=>setActiveTab('chat')} style={{
          background: activeTab==='chat' ? '#0ea5e9' : '#1f2937', color: activeTab==='chat' ? '#0b1120':'#e5e7eb',
          border:'none', borderRadius:8, padding:'6px 10px', cursor:'pointer', fontWeight:600
        }}>Chat</button>
        <button onClick={()=>setActiveTab('llm')} style={{
          background: activeTab==='llm' ? '#0ea5e9' : '#1f2937', color: activeTab==='llm' ? '#0b1120':'#e5e7eb',
          border:'none', borderRadius:8, padding:'6px 10px', cursor:'pointer', fontWeight:600
        }}>LLM Input</button>
      </div>

      {activeTab === 'chat' ? (
        <>
          <div ref={scrollerRef} style={{flex:1,overflow:'auto',padding:16}}>
            {turns.map((t, i) => (
              <div key={i} style={{
                maxWidth:800, margin:'0 auto 12px', padding:'12px 14px', borderRadius:10, whiteSpace:'pre-wrap', lineHeight:1.35,
                background: t.role === 'user' ? '#1f2937' : '#0ea5e9', color: t.role === 'user' ? '#e5e7eb' : '#0b1120'
              }}>{t.text}</div>
            ))}
          </div>
          <form onSubmit={send} style={{display:'flex',gap:8,padding:12,background:'#111827',borderTop:'1px solid #1f2937'}}>
            <input value={message} onChange={e=>setMessage(e.target.value)} placeholder="Type a message..." style={{flex:1,background:'#0f172a',color:'#e5e7eb',border:'1px solid #1f2937',borderRadius:8,padding:'10px 12px'}} />
            <button disabled={busy} style={{background:'#22c55e',color:'#052e16',border:'none',borderRadius:8,padding:'10px 14px',fontWeight:600,cursor:'pointer',opacity:busy?0.6:1}}>Send</button>
          </form>
        </>
      ) : (
        <div style={{flex:1,overflow:'auto',padding:16,display:'grid',gridTemplateColumns:'1fr 1fr',gap:16}}>
          <div style={{background:'#0f172a',border:'1px solid #1f2937',borderRadius:8,padding:12}}>
            <div style={{fontWeight:600,marginBottom:8}}>Clean text (what we send)</div>
            <pre style={{whiteSpace:'pre-wrap',margin:0}}>{payloadText || 'Send a message to see payload...'}</pre>
          </div>
          <div style={{background:'#0f172a',border:'1px solid #1f2937',borderRadius:8,padding:12}}>
            <div style={{fontWeight:600,marginBottom:8}}>Exact JSON payload</div>
            <pre style={{whiteSpace:'pre-wrap',margin:0}}>{payloadJson ? JSON.stringify(payloadJson, null, 2) : 'Send a message to see payload...'}</pre>
          </div>
        </div>
      )}
    </div>
  )
}


