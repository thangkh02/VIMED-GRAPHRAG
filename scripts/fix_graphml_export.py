import json
import os

notebook_path = "d:/Project/ViMed-GraphRAG/AMG_Improved_Entity_Extraction.ipynb"

# Load the notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find Cell 15 (Export Graph)
# We look for the cell source starting with "# === CELL 15: EXPORT GRAPH ==="
target_cell_index = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and len(cell['source']) > 0 and "# === CELL 15: EXPORT GRAPH ===" in cell['source'][0]:
        target_cell_index = i
        break

if target_cell_index != -1:
    new_source = [
        "# === CELL 15: EXPORT GRAPH ===\n",
        "# Export to various formats\n",
        "\n",
        "# 1. Export to JSON\n",
        "import json\n",
        "from networkx.readwrite import json_graph\n",
        "\n",
        "graph_json = json_graph.node_link_data(G)\n",
        "with open(\"./amg_data/graph_improved.json\", \"w\", encoding=\"utf-8\") as f:\n",
        "    json.dump(graph_json, f, indent=2, ensure_ascii=False)\n",
        "print(\" Exported to: ./amg_data/graph_improved.json\")\n",
        "\n",
        "# 2. Export to GraphML (for Gephi, Cytoscape)\n",
        "# Fix: Convert list attributes to strings for GraphML\n",
        "G_export = G.copy()\n",
        "for node, data in G_export.nodes(data=True):\n",
        "    for k, v in data.items():\n",
        "        if isinstance(v, list):\n",
        "            data[k] = str(v)\n",
        "\n",
        "nx.write_graphml(G_export, \"./amg_data/graph_improved.graphml\")\n",
        "print(\" Exported to: ./amg_data/graph_improved.graphml\")\n",
        "\n",
        "# 3. Export to CSV (edges)\n",
        "import pandas as pd\n",
        "edges_data = []\n",
        "for u, v, data in G.edges(data=True):\n",
        "    edges_data.append({\n",
        "        \"source\": u,\n",
        "        \"target\": v,\n",
        "        \"relation\": data.get('relation', ''),\n",
        "        \"confidence\": data.get('confidence', 0),\n",
        "        \"evidence\": data.get('evidence', '')[:100]\n",
        "    })\n",
        "df_edges = pd.DataFrame(edges_data)\n",
        "df_edges.to_csv(\"./amg_data/graph_edges.csv\", index=False, encoding=\"utf-8-sig\")\n",
        "print(\" Exported to: ./amg_data/graph_edges.csv\")\n",
        "\n",
        "print(\"\\nAll exports complete!\")"
    ]
    nb['cells'][target_cell_index]['source'] = new_source
    
    # Save back to file
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=4, ensure_ascii=False)
    print("Successfully fixed GraphML export in notebook.")
else:
    print("Error: Cell 15 not found.")
