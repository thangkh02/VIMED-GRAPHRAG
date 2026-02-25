import { useState, useRef } from 'react'
import { uploadFiles } from '../api'

export default function UploadPanel() {
    const [files, setFiles] = useState([])
    const [uploading, setUploading] = useState(false)
    const [toast, setToast] = useState(null)
    const [dragActive, setDragActive] = useState(false)
    const fileInputRef = useRef(null)

    const showToast = (type, message) => {
        setToast({ type, message })
        setTimeout(() => setToast(null), 5000)
    }

    const formatSize = (bytes) => {
        if (bytes < 1024) return bytes + ' B'
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
    }

    const handleFiles = (newFiles) => {
        const pdfs = Array.from(newFiles).filter((f) =>
            f.name.toLowerCase().endsWith('.pdf')
        )
        if (pdfs.length === 0) {
            showToast('error', 'Only PDF files are accepted.')
            return
        }
        setFiles((prev) => {
            const existing = new Set(prev.map((f) => f.name))
            const unique = pdfs.filter((f) => !existing.has(f.name))
            return [...prev, ...unique]
        })
    }

    const handleDrop = (e) => {
        e.preventDefault()
        setDragActive(false)
        handleFiles(e.dataTransfer.files)
    }

    const handleDragOver = (e) => {
        e.preventDefault()
        setDragActive(true)
    }

    const handleDragLeave = () => {
        setDragActive(false)
    }

    const removeFile = (idx) => {
        setFiles((prev) => prev.filter((_, i) => i !== idx))
    }

    const handleUpload = async () => {
        if (files.length === 0 || uploading) return
        setUploading(true)

        try {
            const data = await uploadFiles(files)
            showToast('success', data.message || 'Files uploaded successfully!')
            setFiles([])
        } catch (err) {
            showToast('error', err.message)
        } finally {
            setUploading(false)
        }
    }

    return (
        <div className="upload-panel">
            <h2>Upload Documents</h2>
            <p>Upload medical PDF documents for knowledge extraction and graph building</p>

            {/* Dropzone */}
            <div
                className={`dropzone ${dragActive ? 'active' : ''}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current?.click()}
            >
                <span className="dropzone-icon">{'\u{1F4E5}'}</span>
                <h3>Drag & Drop PDF files here</h3>
                <p>or click to browse your files</p>
                <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf"
                    multiple
                    style={{ display: 'none' }}
                    onChange={(e) => handleFiles(e.target.files)}
                />
            </div>

            {/* File list */}
            {files.length > 0 && (
                <div className="file-list">
                    {files.map((file, idx) => (
                        <div className="file-item" key={file.name}>
                            <span className="file-icon">{'\u{1F4D1}'}</span>
                            <div className="file-info">
                                <div className="file-name">{file.name}</div>
                                <div className="file-size">{formatSize(file.size)}</div>
                            </div>
                            <button
                                className="file-remove"
                                onClick={() => removeFile(idx)}
                                title="Remove file"
                            >
                                {'\u2715'}
                            </button>
                        </div>
                    ))}
                </div>
            )}

            {/* Upload button */}
            {files.length > 0 && (
                <button
                    className="upload-btn"
                    onClick={handleUpload}
                    disabled={uploading}
                >
                    {uploading ? (
                        <>
                            <span className="spinner" style={{ display: 'inline-block', marginRight: 8 }} />
                            Uploading...
                        </>
                    ) : (
                        `Upload ${files.length} file${files.length > 1 ? 's' : ''}`
                    )}
                </button>
            )}

            {/* Toast notification */}
            {toast && (
                <div className={`toast ${toast.type}`}>
                    {toast.message}
                </div>
            )}
        </div>
    )
}
