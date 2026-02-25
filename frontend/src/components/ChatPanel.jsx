import { useState, useRef, useEffect } from 'react'
import { searchQuery } from '../api'
import MessageBubble, { TypingIndicator } from './MessageBubble'

export default function ChatPanel() {
    const [messages, setMessages] = useState([])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const messagesEndRef = useRef(null)
    const inputRef = useRef(null)

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages, loading])

    const getTimeString = () => {
        return new Date().toLocaleTimeString('vi-VN', {
            hour: '2-digit',
            minute: '2-digit',
        })
    }

    const handleSend = async () => {
        const query = input.trim()
        if (!query || loading) return

        const userMsg = { role: 'user', content: query, time: getTimeString() }
        setMessages((prev) => [...prev, userMsg])
        setInput('')
        setLoading(true)

        try {
            const data = await searchQuery(query)
            const answer =
                data.results && data.results.length > 0
                    ? data.results.join('\n\n')
                    : 'No results found for your query.'

            const aiMsg = { role: 'ai', content: answer, time: getTimeString() }
            setMessages((prev) => [...prev, aiMsg])
        } catch (err) {
            const errMsg = {
                role: 'ai',
                content: `Error: ${err.message}. Please check that the backend is running.`,
                time: getTimeString(),
            }
            setMessages((prev) => [...prev, errMsg])
        } finally {
            setLoading(false)
            inputRef.current?.focus()
        }
    }

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSend()
        }
    }

    return (
        <div className="chat-panel">
            <div className="chat-header">
                <h2>Medical Q&A</h2>
                <p>Ask questions about medical knowledge — powered by GraphRAG + PubMed</p>
            </div>

            <div className="chat-messages">
                {messages.length === 0 && !loading ? (
                    <div className="chat-empty">
                        <div className="chat-empty-icon">{'\u{1FA7A}'}</div>
                        <h3>Welcome to ViMed GraphRAG</h3>
                        <p>
                            Ask any medical question. The system will combine vector search,
                            knowledge graph reasoning, and PubMed references to provide
                            evidence-based answers.
                        </p>
                    </div>
                ) : (
                    <>
                        {messages.map((msg, idx) => (
                            <MessageBubble
                                key={idx}
                                role={msg.role}
                                content={msg.content}
                                time={msg.time}
                            />
                        ))}
                        {loading && <TypingIndicator />}
                        <div ref={messagesEndRef} />
                    </>
                )}
            </div>

            <div className="chat-input-area">
                <div className="chat-input-wrapper">
                    <textarea
                        ref={inputRef}
                        className="chat-input"
                        placeholder="Type your medical question..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        rows={1}
                        disabled={loading}
                    />
                    <button
                        className="chat-send-btn"
                        onClick={handleSend}
                        disabled={loading || !input.trim()}
                        title="Send message"
                    >
                        {loading ? <span className="spinner" /> : '\u{27A4}'}
                    </button>
                </div>
            </div>
        </div>
    )
}
