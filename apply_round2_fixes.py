"""
Apply Round 2 Fixes to AMG_Improved_Entity_Extraction.ipynb
"""

import json

# Load notebook
with open(r'd:\Project\ViMed-GraphRAG\AMG_Improved_Entity_Extraction.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print(f"Loaded notebook with {len(notebook['cells'])} cells\n")

# ==============================================================================
# FIX 1: CELL 3 - Preserve medical abbreviations
# ==============================================================================

cell_3_source = [
    "# === CELL 3: MEDICAL TEXT NORMALIZATION (FIXED - Preserve Abbreviations) ===\n",
    "\n",
    "MEDICAL_ABBREVIATIONS = {\n",
    "    \"btm\": \"bệnh thận mạn\", \"tha\": \"tăng huyết áp\", \"đtđ\": \"đái tháo đường\",\n",
    "    \"gfr\": \"GFR\", \"egfr\": \"eGFR\", \"ckd\": \"CKD\", \"acei\": \"ACEI\", \"arb\": \"ARB\",\n",
    "    \"hba1c\": \"HbA1c\", \"ldl\": \"LDL\", \"hdl\": \"HDL\"\n",
    "}\n",
    "\n",
    "MEDICAL_SYNONYMS = {\n",
    "    \"bệnh thận mạn\": [\"bệnh thận mãn\", \"suy thận mạn\", \"CKD\"],\n",
    "    \"đái tháo đường\": [\"tiểu đường\", \"đtđ\", \"diabetes\"],\n",
    "    \"tăng huyết áp\": [\"cao huyết áp\", \"tha\"],\n",
    "}\n",
    "\n",
    "def normalize_medical_text(text: str) -> str:\n",
    "    \"\"\"✅ FIX: Preserve medical abbreviations, don't use .title()\"\"\"\n",
    "    if not text: return \"Unknown\"\n",
    "    text = unicodedata.normalize(\"NFC\", text).strip().lower()\n",
    "    text = re.sub(r'\\\\s+', ' ', text)\n",
    "    words = [MEDICAL_ABBREVIATIONS.get(re.sub(r'[^\\\\w]', '', w), w) for w in text.split()]\n",
    "    text = ' '.join(words)\n",
    "    for canonical, variants in MEDICAL_SYNONYMS.items():\n",
    "        for variant in variants:\n",
    "            text = text.replace(variant.lower(), canonical)\n",
    "    text = re.sub(r'\\\\s+', ' ', text).strip()\n",
    "    # ✅ Only capitalize first char, preserve rest (e.g., eGFR, HbA1c)\n",
    "    if text: text = text[0].upper() + text[1:]\n",
    "    return text\n",
    "\n",
    "normalize_text = normalize_medical_text\n",
    "print(\"✅ Normalization ready (preserves abbreviations)\")"
]

for i, cell in enumerate(notebook['cells']):
    if 'source' in cell and any('CELL 3: MEDICAL TEXT NORMALIZATION' in line for line in cell.get('source', [])):
        notebook['cells'][i]['source'] = cell_3_source
        print(f"✅ Fixed Cell 3 (normalization) - index {i}")
        break

# ==============================================================================
# FIX 2 & 3: CELL 8 - Upgrade UNKNOWN + Edge deduplication
# ==============================================================================

cell_8_source = [
    "# === CELL 8: GRAPH HELPERS (FIXED - Upgrade + Dedup) ===\n",
    "\n",
    "def add_entity_to_graph(G, entity: MedicalEntity, page_num: int, chunk_id: int):\n",
    "    \"\"\"✅ FIX 2: Upgrade from UNKNOWN to real type\"\"\"\n",
    "    norm_name = normalize_text(entity.name)\n",
    "    confidence = min(1.0, entity.relevance_score / 10.0)\n",
    "    \n",
    "    if not G.has_node(norm_name):\n",
    "        G.add_node(norm_name, label=entity.name, type=entity.type.upper(), description=entity.description,\n",
    "                   confidence=confidence, relevance_score=entity.relevance_score, pages=[page_num], chunks=[chunk_id])\n",
    "    else:\n",
    "        # ✅ Upgrade from UNKNOWN\n",
    "        if G.nodes[norm_name].get(\"type\") == \"UNKNOWN\":\n",
    "            G.nodes[norm_name][\"type\"] = entity.type.upper()\n",
    "            G.nodes[norm_name][\"label\"] = entity.name\n",
    "            G.nodes[norm_name][\"description\"] = entity.description\n",
    "        old_conf = G.nodes[norm_name].get('confidence', 0)\n",
    "        if confidence > old_conf:\n",
    "            G.nodes[norm_name]['confidence'] = confidence\n",
    "            if G.nodes[norm_name].get(\"type\") != \"UNKNOWN\":\n",
    "                G.nodes[norm_name]['description'] = entity.description\n",
    "        if page_num not in G.nodes[norm_name]['pages']:\n",
    "            G.nodes[norm_name]['pages'].append(page_num)\n",
    "        if chunk_id not in G.nodes[norm_name]['chunks']:\n",
    "            G.nodes[norm_name]['chunks'].append(chunk_id)\n",
    "\n",
    "def edge_exists(G, src, tgt, rel, chunk_id):\n",
    "    \"\"\"✅ FIX 3: Check for duplicate edges\"\"\"\n",
    "    if not G.has_edge(src, tgt): return False\n",
    "    edge_dict = G.get_edge_data(src, tgt)\n",
    "    if edge_dict:\n",
    "        for key, data in edge_dict.items():\n",
    "            if data.get(\"relation\") == rel and data.get(\"chunk\") == chunk_id:\n",
    "                return True\n",
    "    return False\n",
    "\n",
    "def add_relation_to_graph(G, rel: MedicalRelation, page_num: int, chunk_id: int):\n",
    "    \"\"\"✅ FIX 3: Deduplication before adding edge\"\"\"\n",
    "    src = normalize_text(rel.source_name)\n",
    "    tgt = normalize_text(rel.target_name)\n",
    "    rel_type = rel.relation.upper()\n",
    "    if not G.has_node(src):\n",
    "        G.add_node(src, label=rel.source_name, type=\"UNKNOWN\", confidence=0.5, pages=[page_num], chunks=[chunk_id], description=\"\")\n",
    "    if not G.has_node(tgt):\n",
    "        G.add_node(tgt, label=rel.target_name, type=\"UNKNOWN\", confidence=0.5, pages=[page_num], chunks=[chunk_id], description=\"\")\n",
    "    # ✅ Check duplicate\n",
    "    if not edge_exists(G, src, tgt, rel_type, chunk_id):\n",
    "        G.add_edge(src, tgt, relation=rel_type, confidence=min(1.0, rel.confidence_score/10),\n",
    "                   evidence=rel.evidence, page=page_num, chunk=chunk_id)\n",
    "\n",
    "print(\"✅ Graph helpers ready (upgrade + dedup)\")"
]

for i, cell in enumerate(notebook['cells']):
    if 'source' in cell and any('CELL 8: GRAPH HELPER' in line for line in cell.get('source', [])):
        notebook['cells'][i]['source'] = cell_8_source
        print(f"✅ Fixed Cell 8 (helpers) - index {i}")
        break

# ==============================================================================
# FIX 4: CELL 14 - Sample exploration with MultiDiGraph API
# ==============================================================================

cell_14_source = [
    "# === CELL 14: SAMPLE EXPLORATION (MultiDiGraph API) ===\n",
    "\n",
    "sample_entity = \"Bệnh Thận Mạn\"\n",
    "\n",
    "if sample_entity in G:\n",
    "    print(f\"\\\\n🔍 Exploring: {sample_entity}\\\\n\")\n",
    "    node_data = G.nodes[sample_entity]\n",
    "    print(f\"Type: {node_data['type']}\")\n",
    "    print(f\"Description: {node_data.get('description', '')[:200]}...\")\n",
    "    print(f\"Confidence: {node_data['confidence']:.2f}\")\n",
    "    print(f\"Pages: {node_data['pages']}\")\n",
    "    \n",
    "    # ✅ FIX: MultiDiGraph API for outgoing edges\n",
    "    print(\"\\\\n📤 Outgoing Relations:\")\n",
    "    for u, v, key, data in G.out_edges(sample_entity, keys=True, data=True):\n",
    "        print(f\"   → {data['relation']} → {v} (conf: {data.get('confidence', 0):.2f})\")\n",
    "        evidence = data.get('evidence', '')[:100]\n",
    "        if evidence:\n",
    "            print(f\"      Evidence: {evidence}...\")\n",
    "    \n",
    "    # ✅ FIX: MultiDiGraph API for incoming edges\n",
    "    print(\"\\\\n📥 Incoming Relations:\")\n",
    "    for u, v, key, data in G.in_edges(sample_entity, keys=True, data=True):\n",
    "        print(f\"   {u} → {data['relation']} → (conf: {data.get('confidence', 0):.2f})\")\n",
    "else:\n",
    "    print(f\"⚠️ '{sample_entity}' not found. Try: {list(G.nodes())[:10]}\")"
]

for i, cell in enumerate(notebook['cells']):
    if 'source' in cell and any('CELL 14: SAMPLE' in line for line in cell.get('source', [])):
        notebook['cells'][i]['source'] = cell_14_source
        print(f"✅ Fixed Cell 14 (exploration) - index {i}")
        break

# ==============================================================================
# FIX 5: CELL 16 - Graph reasoning with MultiDiGraph API
# ==============================================================================

cell_16_source = [
    "# === CELL 16: GRAPH REASONING (MultiDiGraph Compatible) ===\n",
    "\n",
    "def get_connected_nodes(G, node_name: str, confidence_threshold: float = 0.5):\n",
    "    \"\"\"✅ FIX: MultiDiGraph compatible\"\"\"\n",
    "    connected = []\n",
    "    if node_name in G:\n",
    "        for _, neighbor, key, data in G.out_edges(node_name, keys=True, data=True):\n",
    "            conf = data.get(\"confidence\", 0)\n",
    "            if conf >= confidence_threshold:\n",
    "                connected.append({\"node\": neighbor, \"relation\": data.get(\"relation\"),\n",
    "                                  \"confidence\": conf, \"evidence\": data.get(\"evidence\", \"\"), \"key\": key})\n",
    "    return connected\n",
    "\n",
    "def explore_path(G, start_node: str, max_depth: int = 3, confidence_threshold: float = 0.5):\n",
    "    \"\"\"✅ FIX: MultiDiGraph compatible\"\"\"\n",
    "    paths, visited = [], set()\n",
    "    def dfs(node, path, accumulated_confidence, depth):\n",
    "        if depth > max_depth or node in visited: return\n",
    "        visited.add(node)\n",
    "        if len(path) > 0:\n",
    "            paths.append({'path': path.copy(), 'confidence': accumulated_confidence, 'final_node': node})\n",
    "        for neighbor_data in get_connected_nodes(G, node, confidence_threshold):\n",
    "            neighbor = neighbor_data['node']\n",
    "            new_confidence = accumulated_confidence * neighbor_data['confidence']\n",
    "            if new_confidence >= confidence_threshold:\n",
    "                new_path = path + [(node, neighbor, neighbor_data['relation'])]\n",
    "                dfs(neighbor, new_path, new_confidence, depth + 1)\n",
    "        visited.remove(node)\n",
    "    dfs(start_node, [], 1.0, 0)\n",
    "    return paths\n",
    "\n",
    "def reason_about_entity(G, entity_name: str, context_depth: int = 2):\n",
    "    \"\"\"✅ FIX: MultiDiGraph compatible\"\"\"\n",
    "    if entity_name not in G: return f\"Entity '{entity_name}' not found\"\n",
    "    node_data = G.nodes[entity_name]\n",
    "    context = f\"## {entity_name}\\\\nType: {node_data.get('type')}\\\\nConfidence: {node_data.get('confidence', 0):.2f}\\\\n\\\\n\"\n",
    "    context += \"### Direct Relations:\\\\n\"\n",
    "    for conn in get_connected_nodes(G, entity_name, 0.3)[:5]:\n",
    "        context += f\"- {conn['relation']} → {conn['node']} (conf: {conn['confidence']:.2f})\\\\n\"\n",
    "    if context_depth > 1:\n",
    "        context += \"\\\\n### Reasoning Paths:\\\\n\"\n",
    "        paths = explore_path(G, entity_name, context_depth, 0.3)\n",
    "        top_paths = sorted(paths, key=lambda x: x['confidence'], reverse=True)[:3]\n",
    "        for p in top_paths:\n",
    "            path_str = \" → \".join([f\"{step[0]} [{step[2]}]\" for step in p['path']])\n",
    "            if path_str: context += f\"- {path_str} → {p['final_node']} (conf: {p['confidence']:.2f})\\\\n\"\n",
    "    return context\n",
    "\n",
    "def find_related_entities(G, entity_name: str, top_k: int = 10, min_confidence: float = 0.5):\n",
    "    \"\"\"✅ FIX: MultiDiGraph compatible\"\"\"\n",
    "    if entity_name not in G: return []\n",
    "    entity_scores = {}\n",
    "    for conn in get_connected_nodes(G, entity_name, min_confidence):\n",
    "        neighbor = conn['node']\n",
    "        entity_scores[neighbor] = entity_scores.get(neighbor, 0) + conn['confidence']\n",
    "    for conn1 in get_connected_nodes(G, entity_name, 0.3):\n",
    "        for conn2 in get_connected_nodes(G, conn1['node'], 0.3):\n",
    "            second_hop = conn2['node']\n",
    "            if second_hop != entity_name and second_hop not in entity_scores:\n",
    "                propagated = conn1['confidence'] * conn2['confidence'] * 0.5\n",
    "                if propagated >= min_confidence:\n",
    "                    entity_scores[second_hop] = propagated\n",
    "    sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]\n",
    "    return [{'entity': e, 'type': G.nodes[e].get('type'), 'score': s} for e, s in sorted_entities]\n",
    "\n",
    "print(\"✅ Graph Reasoning (MultiDiGraph compatible)\")"
]

for i, cell in enumerate(notebook['cells']):
    if 'source' in cell and any('CELL 16: GRAPH REASONING' in line for line in cell.get('source', [])):
        notebook['cells'][i]['source'] = cell_16_source
        print(f"✅ Fixed Cell 16 (reasoning) - index {i}")
        break

# Save fixed notebook
with open(r'd:\Project\ViMed-GraphRAG\AMG_Improved_Entity_Extraction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4, ensure_ascii=False)

print("\n" + "="*60)
print("✅ ROUND 2 FIXES APPLIED!")
print("="*60)
print("\nFixed:")
print("1. ✅ Cell 3: Preserve medical abbreviations (no .title())")
print("2. ✅ Cell 8: Upgrade UNKNOWN nodes + edge deduplication")
print("3. ✅ Cell 14: MultiDiGraph API for sample exploration")
print("4. ✅ Cell 16: MultiDiGraph API for graph reasoning")
