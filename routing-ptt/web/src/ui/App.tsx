import React, { useEffect, useRef, useState } from 'react'

type Turn = { role: 'user' | 'assistant', text: string }

export default function App() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState(false)
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
    </div>
  )
}


