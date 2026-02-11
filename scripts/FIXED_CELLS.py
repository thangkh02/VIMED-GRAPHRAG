# ==============================================================================
# CRITICAL FIXES FOR AMG_Improved_Entity_Extraction.ipynb
# Thay thế các cells tương ứng bằng code bên dưới
# ==============================================================================

# ==============================================================================
# FIX 1 & 2: CELL 6 - VALIDATION & INVERSE RELATIONS (FIXED)
# ==============================================================================

# Valid relation types - BỔ SUNG INVERSE TYPES VÀO WHITELIST
VALID_RELATION_TYPES = {
    # Original 12 types
    "CAUSES", "TREATS", "PREVENTS", "DIAGNOSES",  # TIER 1
    "SYMPTOM_OF", "COMPLICATION_OF", "SIDE_EFFECT_OF", "INCREASES_RISK",  # TIER 2
    "INTERACTS_WITH", "WORSENS", "INDICATES",  # TIER 3
    "RELATED_TO",  # TIER 4
    
    # ✅ FIX 1: Thêm inverse types vào whitelist
    "CAUSED_BY", "TREATED_BY", "PREVENTED_BY", "DIAGNOSED_BY",
    "HAS_SYMPTOM", "HAS_COMPLICATION", "HAS_SIDE_EFFECT", 
    "RISK_INCREASED_BY", "WORSENED_BY", "INDICATED_BY"
}

# Inverse relation mapping for bidirectional graph
INVERSE_RELATIONS = {
    "CAUSES": "CAUSED_BY",
    "TREATS": "TREATED_BY",
    "PREVENTS": "PREVENTED_BY",
    "DIAGNOSES": "DIAGNOSED_BY",
    "SYMPTOM_OF": "HAS_SYMPTOM",
    "COMPLICATION_OF": "HAS_COMPLICATION",
    "SIDE_EFFECT_OF": "HAS_SIDE_EFFECT",
    "INCREASES_RISK": "RISK_INCREASED_BY",
    "WORSENS": "WORSENED_BY",
    "INDICATES": "INDICATED_BY",
    # INTERACTS_WITH and RELATED_TO are symmetric - no inverse needed
}

SYMMETRIC_RELATIONS = {"INTERACTS_WITH", "RELATED_TO"}

def validate_entity(entity: MedicalEntity) -> bool:
    """Validate if entity is valid medical entity"""
    # Skip empty or too short
    if not entity.name or len(entity.name) < 2:
        return False
    
    # Skip administrative content
    admin_patterns = [
        r'quyết định', r'văn bản', r'bộ y tế', r'trang \d+',
        r'điều \d+', r'khoản \d+', r'mục \d+', r'phụ lục'
    ]
    for pattern in admin_patterns:
        if re.search(pattern, entity.name.lower()):
            return False
    
    # Valid entity types
    valid_types = {
        'DISEASE', 'DRUG', 'SYMPTOM', 'TEST', 'ANATOMY',
        'TREATMENT', 'PROCEDURE', 'RISK_FACTOR', 'LAB_VALUE'
    }
    if entity.type.upper() not in valid_types:
        return False
    
    return True

def validate_relation(relation: MedicalRelation) -> bool:
    """Validate if relation is valid"""
    # Check confidence threshold (increased from 5 to 6)
    if relation.confidence_score < 6:
        return False
    
    # Check valid relation type
    if relation.relation.upper() not in VALID_RELATION_TYPES:
        return False
    
    # Check source and target are valid
    if len(relation.source_name) < 2 or len(relation.target_name) < 2:
        return False
    
    # Avoid self-loops
    if normalize_text(relation.source_name) == normalize_text(relation.target_name):
        return False
    
    return True

def generate_inverse_relations(relations: List[MedicalRelation]) -> List[MedicalRelation]:
    """Generate inverse relations for bidirectional graph"""
    inverse_rels = []
    
    for rel in relations:
        rel_type = rel.relation.upper()
        
        # Skip symmetric relations
        if rel_type in SYMMETRIC_RELATIONS:
            continue
        
        # Generate inverse if mapping exists
        if rel_type in INVERSE_RELATIONS:
            # ✅ FIX 2: Đảm bảo confidence >= 6
            inverse_confidence = max(6, rel.confidence_score - 1)
            
            inverse_rel = MedicalRelation(
                source_name=rel.target_name,
                target_name=rel.source_name,
                relation=INVERSE_RELATIONS[rel_type],
                confidence_score=inverse_confidence,
                evidence=f"Inverse of: {rel.evidence}"
            )
            inverse_rels.append(inverse_rel)
    
    return inverse_rels

print("✅ Validation & Inverse Relations ready (FIXED)")


# ==============================================================================
# FIX 3: CELL 8 - GRAPH HELPER FUNCTIONS (FIXED - MultiDiGraph)
# ==============================================================================

def add_entity_to_graph(G, entity: MedicalEntity, page_num: int, chunk_id: int):
    """Add entity with deduplication & normalization"""
    norm_name = normalize_text(entity.name)
    confidence = min(1.0, entity.relevance_score / 10.0)
    
    if not G.has_node(norm_name):
        G.add_node(
            norm_name,
            label=entity.name,
            type=entity.type.upper(),
            description=entity.description,
            confidence=confidence,
            relevance_score=entity.relevance_score,
            pages=[page_num],
            chunks=[chunk_id]
        )
    else:
        # Update if higher confidence
        old_conf = G.nodes[norm_name].get('confidence', 0)
        if confidence > old_conf:
            G.nodes[norm_name]['confidence'] = confidence
            G.nodes[norm_name]['description'] = entity.description
        
        # Add page/chunk tracking
        if page_num not in G.nodes[norm_name]['pages']:
            G.nodes[norm_name]['pages'].append(page_num)
        if chunk_id not in G.nodes[norm_name]['chunks']:
            G.nodes[norm_name]['chunks'].append(chunk_id)

def add_relation_to_graph(G, rel: MedicalRelation, page_num: int, chunk_id: int):
    """
    Add relation to graph
    ✅ FIX 3: Sử dụng MultiDiGraph để lưu nhiều relations giữa cùng 2 nodes
    """
    src = normalize_text(rel.source_name)
    tgt = normalize_text(rel.target_name)
    
    # Đảm bảo cả 2 nodes tồn tại
    if not G.has_node(src):
        # Tạo placeholder node nếu thiếu
        G.add_node(src, label=rel.source_name, type="UNKNOWN", 
                   confidence=0.5, pages=[page_num], chunks=[chunk_id])
    
    if not G.has_node(tgt):
        # Tạo placeholder node nếu thiếu
        G.add_node(tgt, label=rel.target_name, type="UNKNOWN",
                   confidence=0.5, pages=[page_num], chunks=[chunk_id])
    
    # ✅ MultiDiGraph cho phép nhiều edges giữa cùng 2 nodes
    # Mỗi edge là một instance riêng với relation type & evidence riêng
    G.add_edge(
        src, tgt,
        relation=rel.relation.upper(),
        confidence=min(1.0, rel.confidence_score / 10),
        evidence=rel.evidence,
        page=page_num,
        chunk=chunk_id
    )

print("✅ Graph helper functions ready (MultiDiGraph - FIXED)")


# ==============================================================================
# FIX 3: CELL 10 - LOAD OR CREATE GRAPH (FIXED - MultiDiGraph)
# ==============================================================================

G, last_chunk_id = checkpoint_manager.load()

if G is None:
    print("⚠️ No checkpoint found. Creating new graph...")
    # ✅ FIX 3: Dùng MultiDiGraph thay vì DiGraph
    G = nx.MultiDiGraph()
    start_chunk = 0
else:
    # Resume from last checkpoint
    start_chunk = (last_chunk_id + 1) if last_chunk_id is not None else 0
    print(f"▶️  Resuming from chunk {start_chunk}")

print(f"\n📊 Current state: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


# ==============================================================================
# FIX 4 & 5: CELL 12 - MAIN EXTRACTION LOOP (FIXED)
# ==============================================================================

extractor = ImprovedAMGExtractor(api_manager)

print("🚀 STARTING EXTRACTION WITH IMPROVEMENTS...")
print("   ✅ 12 Relation Types (TIER Priority)")
print("   ✅ Validation (conf >= 6, valid types, no self-loop)")
print("   ✅ Inverse Relations (bidirectional graph)")
print("   ✅ Smart Checkpoint (chunk_id tracking)")
print("   ✅ MultiDiGraph (multi-relations supported)\n")

total_entities_added = 0
total_relations_added = 0

# ✅ FIX 5: Track last processed chunk correctly
last_processed_chunk = start_chunk - 1  # Start before first chunk

for i, chunk in enumerate(chunks_to_process, start=start_chunk):
    chunk_text = chunk.page_content
    page_num = chunk.metadata.get('page', 0)
    
    # Extract
    result = extractor.extract(chunk_text)
    
    # ✅ FIX 4: Chỉ skip khi CẢ entities VÀ relations đều rỗng
    if not result or (not result.entities and not result.relations):
        print(f"   Chunk {i+1}/{len(chunks)}: --")
        continue
    
    # Add entities to graph
    for entity in result.entities:
        add_entity_to_graph(G, entity, page_num, i)
    
    # Add relations to graph
    for relation in result.relations:
        add_relation_to_graph(G, relation, page_num, i)
    
    # Track stats
    num_entities = len(result.entities)
    num_relations = len(result.relations)
    total_entities_added += num_entities
    total_relations_added += num_relations
    
    # Calculate average relevance score
    avg_score = sum(e.relevance_score for e in result.entities) / num_entities if num_entities > 0 else 0
    
    print(f"   Chunk {i+1}/{len(chunks)}: +{num_entities} entities, +{num_relations} relations (avg: {avg_score:.1f})")
    
    # ✅ FIX 5: Update last_processed_chunk
    last_processed_chunk = i
    
    # Save checkpoint every 20 chunks
    if (i + 1) % 20 == 0:
        checkpoint_manager.save(G, i, len(chunks))

# ✅ FIX 5: Final checkpoint với chunk cuối thực sự đã xử lý
checkpoint_manager.save(G, last_processed_chunk, len(chunks))

print(f"\n{'='*60}")
print("✅ EXTRACTION COMPLETE!")
print(f"{'='*60}")
print(f"📊 Final Stats:")
print(f"   - Total Entities: {G.number_of_nodes()}")
print(f"   - Total Relations: {G.number_of_edges()}")
print(f"   - Entities Added This Run: {total_entities_added}")
print(f"   - Relations Added This Run: {total_relations_added}")
print(f"   - Chunks Processed: {last_processed_chunk - start_chunk + 1}")
print(f"   - Last Chunk ID: {last_processed_chunk}")


# ==============================================================================
# CELL 9 - CHECKPOINT MANAGER (Cập nhật type hint cho MultiDiGraph)
# ==============================================================================

class CheckpointManager:
    def __init__(self, checkpoint_dir="./amg_data"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.graph_path = os.path.join(checkpoint_dir, "graph_improved.pkl")
        self.meta_path = os.path.join(checkpoint_dir, "checkpoint_meta.json")
    
    def save(self, G: nx.MultiDiGraph, chunk_id: int, total_chunks: int):
        """Save graph and metadata"""
        # Save graph
        with open(self.graph_path, "wb") as f:
            pickle.dump(G, f)
        
        # Save metadata
        meta = {
            "last_chunk_id": chunk_id,
            "total_chunks": total_chunks,
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Checkpoint: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (chunk {chunk_id})")
    
    def load(self) -> tuple[Optional[nx.MultiDiGraph], Optional[int]]:
        """Load graph and last chunk_id"""
        graph = None
        last_chunk_id = None
        
        # Load graph
        if os.path.exists(self.graph_path):
            with open(self.graph_path, "rb") as f:
                graph = pickle.load(f)
            print(f"✅ Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Load metadata
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            last_chunk_id = meta.get("last_chunk_id")
            print(f"✅ Last checkpoint: chunk {last_chunk_id} at {meta.get('timestamp')}")
        
        return graph, last_chunk_id

checkpoint_manager = CheckpointManager()
print("✅ Checkpoint Manager ready (MultiDiGraph support)")


# ==============================================================================
# CELL 13 - GRAPH STATISTICS (Cập nhật cho MultiDiGraph)
# ==============================================================================

print("\n📊 DETAILED GRAPH STATISTICS\n")
print(f"Total Nodes: {G.number_of_nodes()}")
print(f"Total Edges: {G.number_of_edges()}")

# Entity type distribution
from collections import Counter
entity_types = Counter([G.nodes[n]['type'] for n in G.nodes()])
print("\nEntity Types:")
for etype, count in entity_types.most_common():
    print(f"   {etype}: {count}")

# ✅ MultiDiGraph: Iterate edges with keys
relation_types = Counter([data['relation'] for u, v, key, data in G.edges(keys=True, data=True)])
print("\nRelation Types:")
for rtype, count in relation_types.most_common():
    inverse_marker = " (inverse)" if rtype.endswith("_BY") else ""
    print(f"   {rtype}{inverse_marker}: {count}")

# Top entities by degree
degrees = dict(G.degree())
top_entities = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 Most Connected Entities:")
for entity, degree in top_entities:
    etype = G.nodes[entity]['type']
    print(f"   {entity} ({etype}): {degree} connections")
