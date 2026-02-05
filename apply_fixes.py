"""
Script to apply all critical fixes to AMG_Improved_Entity_Extraction.ipynb
"""

import json

# Load notebook
with open(r'd:\Project\ViMed-GraphRAG\AMG_Improved_Entity_Extraction.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# ==============================================================================
# FIX 1 & 2: CELL 6 - Add inverse types to whitelist, ensure confidence >= 6
# ==============================================================================

cell_6_source = [
    "# === CELL 6: VALIDATION & INVERSE RELATIONS (FIXED) ===\n",
    "\n",
    "# Valid relation types - ✅ BỔ SUNG INVERSE TYPES VÀO WHITELIST\n",
    "VALID_RELATION_TYPES = {\n",
    "    # Original 12 types\n",
    "    \"CAUSES\", \"TREATS\", \"PREVENTS\", \"DIAGNOSES\",  # TIER 1\n",
    "    \"SYMPTOM_OF\", \"COMPLICATION_OF\", \"SIDE_EFFECT_OF\", \"INCREASES_RISK\",  # TIER 2\n",
    "    \"INTERACTS_WITH\", \"WORSENS\", \"INDICATES\",  # TIER 3\n",
    "    \"RELATED_TO\",  # TIER 4\n",
    "    \n",
    "    # ✅ FIX 1: Thêm inverse types vào whitelist\n",
    "    \"CAUSED_BY\", \"TREATED_BY\", \"PREVENTED_BY\", \"DIAGNOSED_BY\",\n",
    "    \"HAS_SYMPTOM\", \"HAS_COMPLICATION\", \"HAS_SIDE_EFFECT\", \n",
    "    \"RISK_INCREASED_BY\", \"WORSENED_BY\", \"INDICATED_BY\"\n",
    "}\n",
    "\n",
    "# Inverse relation mapping\n",
    "INVERSE_RELATIONS = {\n",
    "    \"CAUSES\": \"CAUSED_BY\",\n",
    "    \"TREATS\": \"TREATED_BY\",\n",
    "    \"PREVENTS\": \"PREVENTED_BY\",\n",
    "    \"DIAGNOSES\": \"DIAGNOSED_BY\",\n",
    "    \"SYMPTOM_OF\": \"HAS_SYMPTOM\",\n",
    "    \"COMPLICATION_OF\": \"HAS_COMPLICATION\",\n",
    "    \"SIDE_EFFECT_OF\": \"HAS_SIDE_EFFECT\",\n",
    "    \"INCREASES_RISK\": \"RISK_INCREASED_BY\",\n",
    "    \"WORSENS\": \"WORSENED_BY\",\n",
    "    \"INDICATES\": \"INDICATED_BY\",\n",
    "}\n",
    "\n",
    "SYMMETRIC_RELATIONS = {\"INTERACTS_WITH\", \"RELATED_TO\"}\n",
    "\n",
    "def validate_entity(entity: MedicalEntity) -> bool:\n",
    "    \"\"\"Validate if entity is valid\"\"\"\n",
    "    if not entity.name or len(entity.name) < 2:\n",
    "        return False\n",
    "    admin_patterns = [r'quyết định', r'văn bản', r'bộ y tế', r'trang \\d+', r'điều \\d+', r'khoản \\d+', r'mục \\d+', r'phụ lục']\n",
    "    for pattern in admin_patterns:\n",
    "        if re.search(pattern, entity.name.lower()):\n",
    "            return False\n",
    "    valid_types = {'DISEASE', 'DRUG', 'SYMPTOM', 'TEST', 'ANATOMY', 'TREATMENT', 'PROCEDURE', 'RISK_FACTOR', 'LAB_VALUE'}\n",
    "    if entity.type.upper() not in valid_types:\n",
    "        return False\n",
    "    return True\n",
    "\n",
    "def validate_relation(relation: MedicalRelation) -> bool:\n",
    "    \"\"\"Validate if relation is valid\"\"\"\n",
    "    if relation.confidence_score < 6:\n",
    "        return False\n",
    "    if relation.relation.upper() not in VALID_RELATION_TYPES:\n",
    "        return False\n",
    "    if len(relation.source_name) < 2 or len(relation.target_name) < 2:\n",
    "        return False\n",
    "    if normalize_text(relation.source_name) == normalize_text(relation.target_name):\n",
    "        return False\n",
    "    return True\n",
    "\n",
    "def generate_inverse_relations(relations: List[MedicalRelation]) -> List[MedicalRelation]:\n",
    "    \"\"\"Generate inverse relations\"\"\"\n",
    "    inverse_rels = []\n",
    "    for rel in relations:\n",
    "        rel_type = rel.relation.upper()\n",
    "        if rel_type in SYMMETRIC_RELATIONS:\n",
    "            continue\n",
    "        if rel_type in INVERSE_RELATIONS:\n",
    "            # ✅ FIX 2: Đảm bảo confidence >= 6\n",
    "            inverse_confidence = max(6, rel.confidence_score - 1)\n",
    "            inverse_rel = MedicalRelation(\n",
    "                source_name=rel.target_name,\n",
    "                target_name=rel.source_name,\n",
    "                relation=INVERSE_RELATIONS[rel_type],\n",
    "                confidence_score=inverse_confidence,\n",
    "                evidence=f\"Inverse of: {rel.evidence}\"\n",
    "            )\n",
    "            inverse_rels.append(inverse_rel)\n",
    "    return inverse_rels\n",
    "\n",
    "print(\"✅ Validation & Inverse Relations ready (FIXED)\")"
]

# Update Cell 6
for i, cell in enumerate(notebook['cells']):
    if 'source' in cell and any('CELL 6: VALIDATION' in line for line in cell['source']):
        notebook['cells'][i]['source'] = cell_6_source
        print(f"✅ Fixed Cell 6 (index {i})")
        break

# ==============================================================================
# FIX 3: CELL 8 - MultiDiGraph + placeholder nodes
# ==============================================================================

cell_8_source = [
    "# === CELL 8: GRAPH HELPER FUNCTIONS (FIXED - MultiDiGraph) ===\n",
    "\n",
    "def add_entity_to_graph(G, entity: MedicalEntity, page_num: int, chunk_id: int):\n",
    "    \"\"\"Add entity with deduplication & normalization\"\"\"\n",
    "    norm_name = normalize_text(entity.name)\n",
    "    confidence = min(1.0, entity.relevance_score / 10.0)\n",
    "    \n",
    "    if not G.has_node(norm_name):\n",
    "        G.add_node(norm_name, label=entity.name, type=entity.type.upper(), description=entity.description,\n",
    "                   confidence=confidence, relevance_score=entity.relevance_score, pages=[page_num], chunks=[chunk_id])\n",
    "    else:\n",
    "        old_conf = G.nodes[norm_name].get('confidence', 0)\n",
    "        if confidence > old_conf:\n",
    "            G.nodes[norm_name]['confidence'] = confidence\n",
    "            G.nodes[norm_name]['description'] = entity.description\n",
    "        if page_num not in G.nodes[norm_name]['pages']:\n",
    "            G.nodes[norm_name]['pages'].append(page_num)\n",
    "        if chunk_id not in G.nodes[norm_name]['chunks']:\n",
    "            G.nodes[norm_name]['chunks'].append(chunk_id)\n",
    "\n",
    "def add_relation_to_graph(G, rel: MedicalRelation, page_num: int, chunk_id: int):\n",
    "    \"\"\"Add relation - ✅ FIX 3: MultiDiGraph + placeholder nodes\"\"\"\n",
    "    src = normalize_text(rel.source_name)\n",
    "    tgt = normalize_text(rel.target_name)\n",
    "    \n",
    "    # ✅ Tạo placeholder node nếu thiếu\n",
    "    if not G.has_node(src):\n",
    "        G.add_node(src, label=rel.source_name, type=\"UNKNOWN\", confidence=0.5, pages=[page_num], chunks=[chunk_id])\n",
    "    if not G.has_node(tgt):\n",
    "        G.add_node(tgt, label=rel.target_name, type=\"UNKNOWN\", confidence=0.5, pages=[page_num], chunks=[chunk_id])\n",
    "    \n",
    "    # ✅ MultiDiGraph cho phép nhiều edges\n",
    "    G.add_edge(src, tgt, relation=rel.relation.upper(), confidence=min(1.0, rel.confidence_score / 10),\n",
    "               evidence=rel.evidence, page=page_num, chunk=chunk_id)\n",
    "\n",
    "print(\"✅ Graph helpers ready (MultiDiGraph - FIXED)\")"
]

for i, cell in enumerate(notebook['cells']):
    if 'source' in cell and any('CELL 8: GRAPH HELPER' in line for line in cell['source']):
        notebook['cells'][i]['source'] = cell_8_source
        print(f"✅ Fixed Cell 8 (index {i})")
        break

# ==============================================================================
# FIX 3: CELL 9 - Update type hints
# ==============================================================================

cell_9_source = [
    "# === CELL 9: CHECKPOINT MANAGER (MultiDiGraph) ===\n",
    "\n",
    "class CheckpointManager:\n",
    "    def __init__(self, checkpoint_dir=\"./amg_data\"):\n",
    "        self.checkpoint_dir = checkpoint_dir\n",
    "        os.makedirs(checkpoint_dir, exist_ok=True)\n",
    "        self.graph_path = os.path.join(checkpoint_dir, \"graph_improved.pkl\")\n",
    "        self.meta_path = os.path.join(checkpoint_dir, \"checkpoint_meta.json\")\n",
    "    \n",
    "    def save(self, G: nx.MultiDiGraph, chunk_id: int, total_chunks: int):\n",
    "        \"\"\"Save graph and metadata\"\"\"\n",
    "        with open(self.graph_path, \"wb\") as f:\n",
    "            pickle.dump(G, f)\n",
    "        meta = {\"last_chunk_id\": chunk_id, \"total_chunks\": total_chunks, \"num_nodes\": G.number_of_nodes(),\n",
    "                \"num_edges\": G.number_of_edges(), \"timestamp\": datetime.now().isoformat()}\n",
    "        with open(self.meta_path, \"w\", encoding=\"utf-8\") as f:\n",
    "            json.dump(meta, f, indent=2, ensure_ascii=False)\n",
    "        print(f\"   💾 Checkpoint: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (chunk {chunk_id})\")\n",
    "    \n",
    "    def load(self) -> tuple[Optional[nx.MultiDiGraph], Optional[int]]:\n",
    "        \"\"\"Load graph and last chunk_id\"\"\"\n",
    "        graph, last_chunk_id = None, None\n",
    "        if os.path.exists(self.graph_path):\n",
    "            with open(self.graph_path, \"rb\") as f:\n",
    "                graph = pickle.load(f)\n",
    "            print(f\"✅ Loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges\")\n",
    "        if os.path.exists(self.meta_path):\n",
    "            with open(self.meta_path, \"r\", encoding=\"utf-8\") as f:\n",
    "                meta = json.load(f)\n",
    "            last_chunk_id = meta.get(\"last_chunk_id\")\n",
    "            print(f\"✅ Last checkpoint: chunk {last_chunk_id} at {meta.get('timestamp')}\")\n",
    "        return graph, last_chunk_id\n",
    "\n",
    "checkpoint_manager = CheckpointManager()\n",
    "print(\"✅ Checkpoint Manager ready (MultiDiGraph)\")"
]

for i, cell in enumerate(notebook['cells']):
    if 'source' in cell and any('CELL 9: CHECKPOINT MANAGER' in line for line in cell['source']):
        notebook['cells'][i]['source'] = cell_9_source
        print(f"✅ Fixed Cell 9 (index {i})")
        break

# ==============================================================================
# FIX 3: CELL 10 - MultiDiGraph creation
# ==============================================================================

cell_10_source = [
    "# === CELL 10: LOAD OR CREATE GRAPH (MultiDiGraph) ===\n",
    "\n",
    "G, last_chunk_id = checkpoint_manager.load()\n",
    "\n",
    "if G is None:\n",
    "    print(\"⚠️ No checkpoint. Creating new graph...\")\n",
    "    # ✅ FIX 3: MultiDiGraph\n",
    "    G = nx.MultiDiGraph()\n",
    "    start_chunk = 0\n",
    "else:\n",
    "    start_chunk = (last_chunk_id + 1) if last_chunk_id is not None else 0\n",
    "    print(f\"▶️  Resuming from chunk {start_chunk}\")\n",
    "\n",
    "print(f\"\\n📊 Current: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\")"
]

for i, cell in enumerate(notebook['cells']):
    if 'source' in cell and any('CELL 10: LOAD OR CREATE GRAPH' in line for line in cell['source']):
        notebook['cells'][i]['source'] = cell_10_source
        print(f"✅ Fixed Cell 10 (index {i})")
        break

# ==============================================================================
# FIX 4 & 5: CELL 12 - Fix skip logic and checkpoint tracking
# ==============================================================================

cell_12_source = [
    "# === CELL 12: MAIN EXTRACTION LOOP (FIXED) ===\n",
    "\n",
    "extractor = ImprovedAMGExtractor(api_manager)\n",
    "\n",
    "print(\"🚀 EXTRACTION WITH IMPROVEMENTS...\")\n",
    "print(\"   ✅ 12 Relation Types (TIER Priority)\")\n",
    "print(\"   ✅ Validation (conf >= 6, valid types, no self-loop)\")\n",
    "print(\"   ✅ Inverse Relations (bidirectional)\")\n",
    "print(\"   ✅ Smart Checkpoint (chunk_id tracking)\")\n",
    "print(\"   ✅ MultiDiGraph (multi-relations supported)\\n\")\n",
    "\n",
    "total_entities_added = 0\n",
    "total_relations_added = 0\n",
    "last_processed_chunk = start_chunk - 1  # ✅ FIX 5: Track correctly\n",
    "\n",
    "for i, chunk in enumerate(chunks_to_process, start=start_chunk):\n",
    "    chunk_text = chunk.page_content\n",
    "    page_num = chunk.metadata.get('page', 0)\n",
    "    result = extractor.extract(chunk_text)\n",
    "    \n",
    "    # ✅ FIX 4: Chỉ skip khi CẢ 2 đều rỗng\n",
    "    if not result or (not result.entities and not result.relations):\n",
    "        print(f\"   Chunk {i+1}/{len(chunks)}: --\")\n",
    "        continue\n",
    "    \n",
    "    for entity in result.entities:\n",
    "        add_entity_to_graph(G, entity, page_num, i)\n",
    "    for relation in result.relations:\n",
    "        add_relation_to_graph(G, relation, page_num, i)\n",
    "    \n",
    "    num_entities = len(result.entities)\n",
    "    num_relations = len(result.relations)\n",
    "    total_entities_added += num_entities\n",
    "    total_relations_added += num_relations\n",
    "    avg_score = sum(e.relevance_score for e in result.entities) / num_entities if num_entities > 0 else 0\n",
    "    print(f\"   Chunk {i+1}/{len(chunks)}: +{num_entities} entities, +{num_relations} relations (avg: {avg_score:.1f})\")\n",
    "    \n",
    "    last_processed_chunk = i  # ✅ FIX 5: Update correctly\n",
    "    if (i + 1) % 20 == 0:\n",
    "        checkpoint_manager.save(G, i, len(chunks))\n",
    "\n",
    "# ✅ FIX 5: Save actual last processed chunk\n",
    "checkpoint_manager.save(G, last_processed_chunk, len(chunks))\n",
    "\n",
    "print(f\"\\n{'='*60}\")\n",
    "print(\"✅ EXTRACTION COMPLETE!\")\n",
    "print(f\"{'='*60}\")\n",
    "print(f\"📊 Final Stats:\")\n",
    "print(f\"   - Nodes: {G.number_of_nodes()}\")\n",
    "print(f\"   - Edges: {G.number_of_edges()}\")\n",
    "print(f\"   - Entities Added: {total_entities_added}\")\n",
    "print(f\"   - Relations Added: {total_relations_added}\")\n",
    "print(f\"   - Chunks Processed: {last_processed_chunk - start_chunk + 1}\")\n",
    "print(f\"   - Last Chunk ID: {last_processed_chunk}\")"
]

for i, cell in enumerate(notebook['cells']):
    if 'source' in cell and any('CELL 12: MAIN EXTRACTION LOOP' in line for line in cell['source']):
        notebook['cells'][i]['source'] = cell_12_source
        print(f"✅ Fixed Cell 12 (index {i})")
        break

# ==============================================================================
# FIX: CELL 13 - Update for MultiDiGraph
# ==============================================================================

cell_13_source = [
    "# === CELL 13: GRAPH STATISTICS (MultiDiGraph) ===\n",
    "\n",
    "print(\"\\n📊 GRAPH STATISTICS\\n\")\n",
    "print(f\"Nodes: {G.number_of_nodes()}\")\n",
    "print(f\"Edges: {G.number_of_edges()}\")\n",
    "\n",
    "from collections import Counter\n",
    "entity_types = Counter([G.nodes[n]['type'] for n in G.nodes()])\n",
    "print(\"\\nEntity Types:\")\n",
    "for etype, count in entity_types.most_common():\n",
    "    print(f\"   {etype}: {count}\")\n",
    "\n",
    "# ✅ MultiDiGraph: use keys=True\n",
    "relation_types = Counter([data['relation'] for u, v, key, data in G.edges(keys=True, data=True)])\n",
    "print(\"\\nRelation Types:\")\n",
    "for rtype, count in relation_types.most_common():\n",
    "    inverse_marker = \" (inverse)\" if rtype.endswith(\"_BY\") else \"\"\n",
    "    print(f\"   {rtype}{inverse_marker}: {count}\")\n",
    "\n",
    "degrees = dict(G.degree())\n",
    "top_entities = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]\n",
    "print(\"\\nTop 10 Most Connected:\")\n",
    "for entity, degree in top_entities:\n",
    "    etype = G.nodes[entity]['type']\n",
    "    print(f\"   {entity} ({etype}): {degree} connections\")"
]

for i, cell in enumerate(notebook['cells']):
    if 'source' in cell and any('CELL 13: GRAPH STATISTICS' in line for line in cell['source']):
        notebook['cells'][i]['source'] = cell_13_source
        print(f"✅ Fixed Cell 13 (index {i})")
        break

# Save fixed notebook
with open(r'd:\Project\ViMed-GraphRAG\AMG_Improved_Entity_Extraction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4, ensure_ascii=False)

print("\n" + "="*60)
print("✅ ALL FIXES APPLIED TO NOTEBOOK!")
print("="*60)
print("\nFixed issues:")
print("1. ✅ Added inverse types to VALID_RELATION_TYPES whitelist")
print("2. ✅ Ensured inverse confidence >= 6")
print("3. ✅ Changed DiGraph → MultiDiGraph (multi-relations)")
print("4. ✅ Fixed chunk skip logic (check both entities AND relations)")
print("5. ✅ Fixed checkpoint tracking (use actual last_processed_chunk)")
