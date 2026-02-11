import json
import sys

# Load existing notebook
with open(r'd:\Project\ViMed-GraphRAG\AMG_Improved_Entity_Extraction.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Cell 16: Graph Reasoning Functions
cell_16 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# === CELL 16: GRAPH REASONING & PATH EXPLORATION ===\n",
        "\n",
        "def get_connected_nodes(G, node_name: str, confidence_threshold: float = 0.5):\n",
        "    \"\"\"Get nodes connected to a given node with confidence above threshold\"\"\"\n",
        "    connected = []\n",
        "    if node_name in G:\n",
        "        for neighbor in G.neighbors(node_name):\n",
        "            edge_data = G[node_name][neighbor]\n",
        "            if edge_data.get('confidence', 0) >= confidence_threshold:\n",
        "                connected.append({\n",
        "                    'node': neighbor,\n",
        "                    'relation': edge_data.get('relation'),\n",
        "                    'confidence': edge_data.get('confidence'),\n",
        "                    'evidence': edge_data.get('evidence')\n",
        "                })\n",
        "    return connected\n",
        "\n",
        "def explore_path(G, start_node: str, max_depth: int = 3, confidence_threshold: float = 0.5):\n",
        "    \"\"\"Explore paths from a starting node with confidence propagation\"\"\"\n",
        "    paths = []\n",
        "    visited = set()\n",
        "    \n",
        "    def dfs(node, path, accumulated_confidence, depth):\n",
        "        if depth > max_depth or node in visited:\n",
        "            return\n",
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
        "    \n",
        "    dfs(start_node, [], 1.0, 0)\n",
        "    return paths\n",
        "\n",
        "def reason_about_entity(G, entity_name: str, context_depth: int = 2):\n",
        "    \"\"\"Generate reasoning context about an entity\"\"\"\n",
        "    if entity_name not in G:\n",
        "        return f\"Entity '{entity_name}' not found\"\n",
        "    \n",
        "    node_data = G.nodes[entity_name]\n",
        "    context = f\"## {entity_name}\\n\"\n",
        "    context += f\"Type: {node_data.get('type')}\\nConfidence: {node_data.get('confidence', 0):.2f}\\n\\n\"\n",
        "    \n",
        "    # Direct connections\n",
        "    context += \"### Direct Relations:\\n\"\n",
        "    for conn in get_connected_nodes(G, entity_name, 0.3)[:5]:\n",
        "        context += f\"- {conn['relation']} → {conn['node']} (conf: {conn['confidence']:.2f})\\n\"\n",
        "    \n",
        "    # Multi-hop paths\n",
        "    if context_depth > 1:\n",
        "        context += \"\\n### Reasoning Paths:\\n\"\n",
        "        paths = explore_path(G, entity_name, context_depth, 0.3)\n",
        "        top_paths = sorted(paths, key=lambda x: x['confidence'], reverse=True)[:3]\n",
        "        for p in top_paths:\n",
        "            path_str = \" → \".join([f\"{step[0]} [{step[2]}]\" for step in p['path']])\n",
        "            if path_str:\n",
        "                context += f\"- {path_str} → {p['final_node']} (conf: {p['confidence']:.2f})\\n\"\n",
        "    return context\n",
        "\n",
        "def find_related_entities(G, entity_name: str, top_k: int = 10, min_confidence: float = 0.5):\n",
        "    \"\"\"Find entities most related to a given entity\"\"\"\n",
        "    if entity_name not in G:\n",
        "        return []\n",
        "    \n",
        "    entity_scores = {}\n",
        "    # Direct connections\n",
        "    for neighbor in G.neighbors(entity_name):\n",
        "        conf = G[entity_name][neighbor].get('confidence', 0)\n",
        "        if conf >= min_confidence:\n",
        "            entity_scores[neighbor] = conf\n",
        "    \n",
        "    # 2-hop (with decay)\n",
        "    for neighbor in G.neighbors(entity_name):\n",
        "        for second_hop in G.neighbors(neighbor):\n",
        "            if second_hop != entity_name and second_hop not in entity_scores:\n",
        "                conf1 = G[entity_name][neighbor].get('confidence', 0)\n",
        "                conf2 = G[neighbor][second_hop].get('confidence', 0)\n",
        "                propagated = conf1 * conf2 * 0.5\n",
        "                if propagated >= min_confidence:\n",
        "                    entity_scores[second_hop] = propagated\n",
        "    \n",
        "    sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]\n",
        "    return [{'entity': e, 'type': G.nodes[e].get('type'), 'score': s} for e, s in sorted_entities]\n",
        "\n",
        "print(\"✅ Graph Reasoning functions ready\")\n",
        "print(\"\\nFunctions: get_connected_nodes, explore_path, reason_about_entity, find_related_entities\")"
    ]
}

# Cell 17: Reasoning Demo
cell_17 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# === CELL 17: EXAMPLE - REASONING DEMO ===\n",
        "\n",
        "# Demo: Reason about a specific entity\n",
        "demo_entity = \"Bệnh Thận Mạn\"  # Change this\n",
        "\n",
        "if demo_entity in G:\n",
        "    print(\"🧠 REASONING ABOUT ENTITY\\n\")\n",
        "    print(reason_about_entity(G, demo_entity, context_depth=2))\n",
        "    \n",
        "    print(\"\\n\" + \"=\"*60)\n",
        "    print(\"🔗 RELATED ENTITIES\\n\")\n",
        "    related = find_related_entities(G, demo_entity, top_k=10, min_confidence=0.3)\n",
        "    for r in related:\n",
        "        print(f\"   {r['entity']} ({r['type']}): {r['score']:.3f}\")\n",
        "    \n",
        "    print(\"\\n\" + \"=\"*60)\n",
        "    print(\"📊 EXPLORING PATHS\\n\")\n",
        "    paths = explore_path(G, demo_entity, max_depth=2, confidence_threshold=0.3)\n",
        "    top_5_paths = sorted(paths, key=lambda x: x['confidence'], reverse=True)[:5]\n",
        "    for i, path_data in enumerate(top_5_paths, 1):\n",
        "        path_str = \" → \".join([f\"{p[0]} [{p[2]}]\" for p in path_data['path']])\n",
        "        if path_str:\n",
        "            print(f\"  {i}. {path_str} → {path_data['final_node']}\")\n",
        "            print(f\"     Confidence: {path_data['confidence']:.3f}\\n\")\n",
        "else:\n",
        "    print(f\"⚠️ '{demo_entity}' not in graph. Try: {list(G.nodes())[:5]}\")"
    ]
}

# Insert cells before the last cell (which is export)
notebook['cells'].insert(-1, cell_16)
notebook['cells'].insert(-1, cell_17)

# Save updated notebook
with open(r'd:\Project\ViMed-GraphRAG\AMG_Improved_Entity_Extraction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4, ensure_ascii=False)

print(f"✅ Added Cell 16 & 17 to notebook")
print(f"📊 Total cells now: {len(notebook['cells'])}")
