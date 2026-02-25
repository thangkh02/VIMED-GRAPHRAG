import { useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatPanel from './components/ChatPanel'
import UploadPanel from './components/UploadPanel'
import GraphPanel from './components/GraphPanel'

const TABS = [
    { id: 'chat', label: 'Medical Q&A', icon: '\u{1F4AC}' },
    { id: 'upload', label: 'Upload Documents', icon: '\u{1F4C4}' },
    { id: 'graph', label: 'Knowledge Graph', icon: '\u{1F578}\uFE0F' },
]

export default function App() {
    const [activeTab, setActiveTab] = useState('chat')

    const renderPanel = () => {
        switch (activeTab) {
            case 'chat': return <ChatPanel />
            case 'upload': return <UploadPanel />
            case 'graph': return <GraphPanel />
            default: return <ChatPanel />
        }
    }

    return (
        <div className="app-layout">
            <Sidebar tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} />
            <main className="main-content">
                <div className="panel-container">
                    {renderPanel()}
                </div>
            </main>
        </div>
    )
}
