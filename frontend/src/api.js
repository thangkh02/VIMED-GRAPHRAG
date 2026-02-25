const BASE_URL = '/api';

/**
 * Send a search query to the backend.
 * POST /search/ -> { results: [string] }
 */
export async function searchQuery(query, topK = 5) {
    const res = await fetch(`${BASE_URL}/search/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: topK }),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Search failed (${res.status})`);
    }
    return res.json();
}

/**
 * Upload PDF files for ingestion.
 * POST /ingest/ -> { message, files }
 */
export async function uploadFiles(files) {
    const formData = new FormData();
    for (const file of files) {
        formData.append('files', file);
    }
    const res = await fetch(`${BASE_URL}/ingest/`, {
        method: 'POST',
        body: formData,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Upload failed (${res.status})`);
    }
    return res.json();
}

/**
 * Get the URL for the graph visualization iframe.
 */
export function getGraphUrl() {
    return `${BASE_URL}/graph/visualize`;
}
