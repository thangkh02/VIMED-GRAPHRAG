export default function MessageBubble({ role, content, time }) {
    const isUser = role === 'user'

    return (
        <div className={`message ${isUser ? 'user' : 'ai'}`}>
            <div className="message-avatar">
                {isUser ? 'U' : 'V'}
            </div>
            <div>
                <div className="message-content">
                    {content}
                </div>
                {time && <div className="message-time">{time}</div>}
            </div>
        </div>
    )
}

export function TypingIndicator() {
    return (
        <div className="message ai">
            <div className="message-avatar">V</div>
            <div className="message-content">
                <div className="typing-indicator">
                    <div className="typing-dot" />
                    <div className="typing-dot" />
                    <div className="typing-dot" />
                </div>
            </div>
        </div>
    )
}
