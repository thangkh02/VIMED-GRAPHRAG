import { useState } from 'react'
import { getGraphUrl } from '../api'

export default function GraphPanel() {
    const [key, setKey] = useState(0)
    const [loaded, setLoaded] = useState(false)
    const [error, setError] = useState(false)

    const graphUrl = getGraphUrl()

    const handleRefresh = () => {
        setLoaded(false)
        setError(false)
        setKey((prev) => prev + 1)
    }

    const handleLoad = () => {
        setLoaded(true)
    }

    const handleError = () => {
        setError(true)
        setLoaded(true)
    }

    return (
        <div className="graph-panel">
            <div className="graph-header">
                <h2>Knowledge Graph</h2>
                <button className="graph-refresh-btn" onClick={handleRefresh}>
                    {'\u21BB'} Refresh
                </button>
            </div>

            {error ? (
                <div className="graph-empty">
                    <span className="graph-empty-icon">{'\u{1F578}\uFE0F'}</span>
                    <h3>No graph available</h3>
                    <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
                        Upload and ingest a document first to build the knowledge graph,
                        then come back here to visualize it.
                    </p>
                    <button className="graph-refresh-btn" onClick={handleRefresh}>
                        Try again
                    </button>
                </div>
            ) : (
                <div className="graph-iframe-container">
                    {!loaded && (
                        <div
                            style={{
                                position: 'absolute',
                                inset: 0,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                            }}
                        >
                            <span className="spinner" />
                        </div>
                    )}
                    <iframe
                        key={key}
                        className="graph-iframe"
                        src={graphUrl}
                        title="Knowledge Graph Visualization"
                        onLoad={handleLoad}
                        onError={handleError}
                        style={{ opacity: loaded ? 1 : 0 }}
                    />
                </div>
            )}
        </div>
    )
}
