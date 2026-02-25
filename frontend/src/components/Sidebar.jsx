export default function Sidebar({ tabs, activeTab, onTabChange }) {
    return (
        <aside className="sidebar">
            <div className="sidebar-logo">
                <div className="sidebar-logo-icon">V</div>
                <div className="sidebar-logo-text">
                    <h1>ViMed GraphRAG</h1>
                    <span>Medical AI Assistant</span>
                </div>
            </div>

            <nav className="sidebar-nav">
                {tabs.map((tab) => (
                    <div
                        key={tab.id}
                        className={`nav-item ${activeTab === tab.id ? 'active' : ''}`}
                        onClick={() => onTabChange(tab.id)}
                    >
                        <span className="nav-icon">{tab.icon}</span>
                        <span>{tab.label}</span>
                    </div>
                ))}
            </nav>

            <div className="sidebar-footer">
                ViMed GraphRAG v1.0 &middot; Self-MedRAG
            </div>
        </aside>
    )
}
