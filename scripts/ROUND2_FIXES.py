"""
ROUND 2 CRITICAL FIXES - MultiDiGraph Compatibility
Apply these fixes to AMG_Improved_Entity_Extraction.ipynb
"""

# ==============================================================================
# FIX 1: CELL 16 - GRAPH REASONING (MULTIDIGRAPH COMPATIBLE)
# ==============================================================================

def get_connected_nodes(G, node_name: str, confidence_threshold: float = 0.5):
    """Get nodes connected to a given node - ✅ MultiDiGraph compatible"""
    connected = []
    if node_name in G:
        # ✅ FIX: Use out_edges with keys=True, data=True
        for _, neighbor, key, data in G.out_edges(node_name, keys=True, data=True):
            conf = data.get("confidence", 0)
            if conf >= confidence_threshold:
                connected.append({
                    "node": neighbor,
                    "relation": data.get("relation"),
                    "confidence": conf,
                    "evidence": data.get("evidence", ""),
                    "key": key  # ✅ Track edge key for MultiDiGraph
                })
    return connected


def explore_path(G, start_node: str, max_depth: int = 3, confidence_threshold: float = 0.5):
    """Explore paths - ✅ MultiDiGraph compatible"""
    paths = []
    visited = set()
    
    def dfs(node, path, accumulated_confidence, depth):
        if depth > max_depth or node in visited:
            return
        visited.add(node)
        if len(path) > 0:
            paths.append({'path': path.copy(), 'confidence': accumulated_confidence, 'final_node': node})
        
        # ✅ Use get_connected_nodes (already MultiDiGraph compatible)
        for neighbor_data in get_connected_nodes(G, node, confidence_threshold):
            neighbor = neighbor_data['node']
            new_confidence = accumulated_confidence * neighbor_data['confidence']
            if new_confidence >= confidence_threshold:
                new_path = path + [(node, neighbor, neighbor_data['relation'])]
                dfs(neighbor, new_path, new_confidence, depth + 1)
        visited.remove(node)
    
    dfs(start_node, [], 1.0, 0)
    return paths


def reason_about_entity(G, entity_name: str, context_depth: int = 2):
    """Generate reasoning context - ✅ MultiDiGraph compatible"""
    if entity_name not in G:
        return f"Entity '{entity_name}' not found"
    
    node_data = G.nodes[entity_name]
    context = f"## {entity_name}\n"
    context += f"Type: {node_data.get('type')}\nConfidence: {node_data.get('confidence', 0):.2f}\n\n"
    
    # Direct connections
    context += "### Direct Relations:\n"
    for conn in get_connected_nodes(G, entity_name, 0.3)[:5]:
        context += f"- {conn['relation']} → {conn['node']} (conf: {conn['confidence']:.2f})\n"
    
    # Multi-hop paths
    if context_depth > 1:
        context += "\n### Reasoning Paths:\n"
        paths = explore_path(G, entity_name, context_depth, 0.3)
        top_paths = sorted(paths, key=lambda x: x['confidence'], reverse=True)[:3]
        for p in top_paths:
            path_str = " → ".join([f"{step[0]} [{step[2]}]" for step in p['path']])
            if path_str:
                context += f"- {path_str} → {p['final_node']} (conf: {p['confidence']:.2f})\n"
    return context


def find_related_entities(G, entity_name: str, top_k: int = 10, min_confidence: float = 0.5):
    """Find related entities - ✅ MultiDiGraph compatible"""
    if entity_name not in G:
        return []
    
    entity_scores = {}
    
    # Direct connections
    for conn in get_connected_nodes(G, entity_name, min_confidence):
        neighbor = conn['node']
        entity_scores[neighbor] = entity_scores.get(neighbor, 0) + conn['confidence']
    
    # 2-hop (with decay)
    for conn1 in get_connected_nodes(G, entity_name, 0.3):
        neighbor = conn1['node']
        for conn2 in get_connected_nodes(G, neighbor, 0.3):
            second_hop = conn2['node']
            if second_hop != entity_name and second_hop not in entity_scores:
                propagated = conn1['confidence'] * conn2['confidence'] * 0.5
                if propagated >= min_confidence:
                    entity_scores[second_hop] = propagated
    
    sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{'entity': e, 'type': G.nodes[e].get('type'), 'score': s} for e, s in sorted_entities]


print("✅ Graph Reasoning functions ready (MultiDiGraph compatible)")


# ==============================================================================
# FIX 2: CELL 14 - SAMPLE EXPLORATION (MULTIDIGRAPH COMPATIBLE)
# ==============================================================================

# Example: Explore a specific entity
sample_entity = "Bệnh Thận Mạn"  # Change this

if sample_entity in G:
    print(f"\n🔍 Exploring: {sample_entity}\n")
    
    # Node info
    node_data = G.nodes[sample_entity]
    print(f"Type: {node_data['type']}")
    print(f"Description: {node_data.get('description', '')[:200]}...")
    print(f"Confidence: {node_data['confidence']:.2f}")
    print(f"Found on pages: {node_data['pages']}")
    
    # ✅ FIX: Outgoing relations using MultiDiGraph API
    print("\n📤 Outgoing Relations:")
    for u, v, key, data in G.out_edges(sample_entity, keys=True, data=True):
        print(f"   → {data['relation']} → {v} (conf: {data.get('confidence', 0):.2f})")
        evidence = data.get('evidence', '')[:100]
        if evidence:
            print(f"      Evidence: {evidence}...")
    
    # ✅ FIX: Incoming relations using MultiDiGraph API
    print("\n📥 Incoming Relations:")
    for u, v, key, data in G.in_edges(sample_entity, keys=True, data=True):
        print(f"   {u} → {data['relation']} → (conf: {data.get('confidence', 0):.2f})")
else:
    print(f"⚠️ Entity '{sample_entity}' not found")
    print(f"   Try: {list(G.nodes())[:10]}")


# ==============================================================================
# FIX 3 & 4: CELL 3 - MEDICAL TEXT NORMALIZATION (PRESERVE ABBREVIATIONS)
# ==============================================================================

MEDICAL_ABBREVIATIONS = {
    "btm": "bệnh thận mạn", "tha": "tăng huyết áp", "đtđ": "đái tháo đường",
    "gfr": "GFR", "egfr": "eGFR",  # ✅ Preserve case for abbreviations
    "ckd": "CKD", "acei": "ACEI", "arb": "ARB",
    "hba1c": "HbA1c", "ldl": "LDL", "hdl": "HDL"
}

MEDICAL_SYNONYMS = {
    "bệnh thận mạn": ["bệnh thận mãn", "suy thận mạn", "CKD"],
    "đái tháo đường": ["tiểu đường", "đtđ", "diabetes"],
    "tăng huyết áp": ["cao huyết áp", "tha"],
}

# ✅ FIX 4: Preserve medical abbreviations, don't use .title()
def normalize_medical_text(text: str) -> str:
    """Normalize text while preserving medical abbreviations"""
    if not text:
        return "Unknown"
    
    # Normalize unicode
    text = unicodedata.normalize("NFC", text).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    
    # Expand abbreviations
    words = []
    for w in text.split():
        clean_w = re.sub(r'[^\w]', '', w)
        expanded = MEDICAL_ABBREVIATIONS.get(clean_w, w)
        words.append(expanded)
    
    text = ' '.join(words)
    
    # Apply synonyms
    for canonical, variants in MEDICAL_SYNONYMS.items():
        for variant in variants:
            text = text.replace(variant.lower(), canonical)
    
    # ✅ FIX: Capitalize first letter only, preserve rest (especially abbreviations)
    text = re.sub(r'\s+', ' ', text).strip()
    if text:
        text = text[0].upper() + text[1:]  # Only capitalize first char
    
    return text

normalize_text = normalize_medical_text
print("✅ Normalization ready (preserves abbreviations)")


# ==============================================================================
# FIX 5: CELL 8 - ADD ENTITY WITH UPGRADE FROM UNKNOWN
# ==============================================================================

def add_entity_to_graph(G, entity: MedicalEntity, page_num: int, chunk_id: int):
    """Add entity with deduplication & upgrade from UNKNOWN"""
    norm_name = normalize_text(entity.name)
    confidence = min(1.0, entity.relevance_score / 10.0)
    
    if not G.has_node(norm_name):
        G.add_node(norm_name, label=entity.name, type=entity.type.upper(), description=entity.description,
                   confidence=confidence, relevance_score=entity.relevance_score, 
                   pages=[page_num], chunks=[chunk_id])
    else:
        # ✅ FIX 2: Upgrade from UNKNOWN to real type
        if G.nodes[norm_name].get("type") == "UNKNOWN":
            G.nodes[norm_name]["type"] = entity.type.upper()
            G.nodes[norm_name]["label"] = entity.name
            G.nodes[norm_name]["description"] = entity.description
        
        # Update if higher confidence
        old_conf = G.nodes[norm_name].get('confidence', 0)
        if confidence > old_conf:
            G.nodes[norm_name]['confidence'] = confidence
            if G.nodes[norm_name].get("type") != "UNKNOWN":  # Don't overwrite real description
                G.nodes[norm_name]['description'] = entity.description
        
        # Add page/chunk tracking
        if page_num not in G.nodes[norm_name]['pages']:
            G.nodes[norm_name]['pages'].append(page_num)
        if chunk_id not in G.nodes[norm_name]['chunks']:
            G.nodes[norm_name]['chunks'].append(chunk_id)


# ==============================================================================
# FIX 6: CELL 8 - EDGE DEDUPLICATION
# ==============================================================================

def edge_exists(G, src, tgt, rel, chunk_id):
    """Check if edge already exists with same relation and chunk"""
    if not G.has_edge(src, tgt):
        return False
    
    # ✅ MultiDiGraph: Check all edges between src and tgt
    edge_dict = G.get_edge_data(src, tgt)
    if edge_dict:
        for key, data in edge_dict.items():
            if data.get("relation") == rel and data.get("chunk") == chunk_id:
                return True
    return False


def add_relation_to_graph(G, rel: MedicalRelation, page_num: int, chunk_id: int):
    """Add relation with deduplication"""
    src = normalize_text(rel.source_name)
    tgt = normalize_text(rel.target_name)
    rel_type = rel.relation.upper()
    
    # Create placeholder nodes if needed
    if not G.has_node(src):
        G.add_node(src, label=rel.source_name, type="UNKNOWN", confidence=0.5, 
                   pages=[page_num], chunks=[chunk_id], description="")
    if not G.has_node(tgt):
        G.add_node(tgt, label=rel.target_name, type="UNKNOWN", confidence=0.5,
                   pages=[page_num], chunks=[chunk_id], description="")
    
    # ✅ FIX 3: Check for duplicates before adding
    if not edge_exists(G, src, tgt, rel_type, chunk_id):
        G.add_edge(src, tgt, relation=rel_type, 
                   confidence=min(1.0, rel.confidence_score / 10),
                   evidence=rel.evidence, page=page_num, chunk=chunk_id)


print("✅ Graph helpers ready (deduplication + upgrade)")


# ==============================================================================
# FIX 7: CELL 9 - CHECKPOINT WITH FINGERPRINT
# ==============================================================================

import hashlib

class CheckpointManager:
    def __init__(self, checkpoint_dir="./amg_data", pdf_path="", chunk_size=512, model="llama-3.3-70b-versatile"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.graph_path = os.path.join(checkpoint_dir, "graph_improved.pkl")
        self.meta_path = os.path.join(checkpoint_dir, "checkpoint_meta.json")
        
        # ✅ FIX 7: Calculate fingerprint
        self.fingerprint = self._calculate_fingerprint(pdf_path, chunk_size, model)
    
    def _calculate_fingerprint(self, pdf_path, chunk_size, model):
        """Generate fingerprint to detect config changes"""
        fingerprint_str = f"{pdf_path}|{chunk_size}|{model}"
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:8]
    
    def save(self, G: nx.MultiDiGraph, chunk_id: int, total_chunks: int):
        """Save graph and metadata with fingerprint"""
        with open(self.graph_path, "wb") as f:
            pickle.dump(G, f)
        
        meta = {
            "last_chunk_id": chunk_id,
            "total_chunks": total_chunks,
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "timestamp": datetime.now().isoformat(),
            "fingerprint": self.fingerprint  # ✅ Save fingerprint
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Checkpoint: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (chunk {chunk_id})")
    
    def load(self) -> tuple[Optional[nx.MultiDiGraph], Optional[int]]:
        """Load graph with fingerprint validation"""
        graph, last_chunk_id = None, None
        
        if os.path.exists(self.graph_path):
            with open(self.graph_path, "rb") as f:
                graph = pickle.load(f)
            print(f"✅ Loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            # ✅ FIX 7: Validate fingerprint
            saved_fingerprint = meta.get("fingerprint")
            if saved_fingerprint != self.fingerprint:
                print(f"⚠️ WARNING: Config mismatch!")
                print(f"   Saved fingerprint: {saved_fingerprint}")
                print(f"   Current fingerprint: {self.fingerprint}")
                print(f"   → Checkpoint may be from different PDF/config")
                # Still load but warn user
            
            last_chunk_id = meta.get("last_chunk_id")
            print(f"✅ Last checkpoint: chunk {last_chunk_id} at {meta.get('timestamp')}")
        
        return graph, last_chunk_id

# ✅ Usage: Pass PDF_PATH and config
checkpoint_manager = CheckpointManager(
    checkpoint_dir="./amg_data",
    pdf_path=PDF_PATH,
    chunk_size=512,
    model="llama-3.3-70b-versatile"
)
print("✅ Checkpoint Manager ready (with fingerprint validation)")
