import json
import os

notebook_path = "d:/Project/ViMed-GraphRAG/AMG_Improved_Entity_Extraction.ipynb"

# Check if file exists
if not os.path.exists(notebook_path):
    print(f"Error: {notebook_path} not found")
    exit(1)

# Load the notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Define the new cell
new_cell_source = [
    "# === CELL 18: VISUALIZATION WITH PYVIS (COLORIZED) ===\n",
    "from pyvis.network import Network\n",
    "import IPython.display\n",
    "\n",
    "def visualize_graph(G, output_file=\"amg_rag_graph.html\"):\n",
    "    print(f\"Visualizing {G.number_of_nodes()} nodes and {G.number_of_edges()} edges...\")\n",
    "    \n",
    "    net = Network(height=\"750px\", width=\"100%\", bgcolor=\"#222222\", font_color=\"white\", select_menu=True, filter_menu=True)\n",
    "    # Force Atlas 2 Based is good for large networks\n",
    "    net.force_atlas_2based()\n",
    "    \n",
    "    # Define intuitive colors for medical entities\n",
    "    color_map = {\n",
    "        \"DISEASE\": \"#FF6B6B\",       # Red/Salmon\n",
    "        \"DRUG\": \"#4ECDC4\",          # Teal/Green\n",
    "        \"SYMPTOM\": \"#FFE66D\",       # Yellow\n",
    "        \"TEST\": \"#1A535C\",          # Dark Cyan\n",
    "        \"ANATOMY\": \"#FF9F1C\",       # Orange\n",
    "        \"TREATMENT\": \"#2B2D42\",     # Dark Blue\n",
    "        \"PROCEDURE\": \"#8D99AE\",     # Grey Blue\n",
    "        \"RISK_FACTOR\": \"#EF233C\",   # Red\n",
    "        \"LAB_VALUE\": \"#219EBC\",     # Blue\n",
    "        \"UNKNOWN\": \"#cccccc\"\n",
    "    }\n",
    "    \n",
    "    # Add nodes with custom visual properties\n",
    "    for node, data in G.nodes(data=True):\n",
    "        node_type = data.get(\"type\", \"UNKNOWN\")\n",
    "        color = color_map.get(node_type, \"#999999\")\n",
    "        \n",
    "        # Size based on degree (importance)\n",
    "        degree = G.degree(node)\n",
    "        size = 15 + (degree * 2)\n",
    "        \n",
    "        title_html = f\"<b>{node}</b><br>Type: {node_type}<br>Connections: {degree}\"\n",
    "        if \"description\" in data and data[\"description\"]:\n",
    "            title_html += f\"<br><i>{data['description'][:100]}...</i>\"\n",
    "\n",
    "        net.add_node(node, label=node, title=title_html, color=color, value=size, group=node_type)\n",
    "    \n",
    "    # Add edges\n",
    "    for u, v, data in G.edges(data=True):\n",
    "        rel_type = data.get(\"relation\", \"RELATED_TO\")\n",
    "        confidence = data.get(\"confidence\", 0.5)\n",
    "        width = confidence * 3  # Thicker lines for higher confidence\n",
    "        \n",
    "        net.add_edge(u, v, title=f\"{rel_type} (conf: {confidence:.2f})\", label=rel_type, width=width, color=\"#aaaaaa\")\n",
    "    \n",
    "    # Save and display\n",
    "    net.show_buttons(filter_=['physics'])\n",
    "    net.save_graph(output_file)\n",
    "    print(f\" Graph saved to {output_file}\")\n",
    "    return output_file\n",
    "\n",
    "html_file = visualize_graph(G, \"amg_rag_graph.html\")\n",
    "\n",
    "# Display in notebook\n",
    "IPython.display.HTML(filename=html_file)"
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": new_cell_source
}

# Append the new cell
nb['cells'].append(new_cell)

# Save back to file
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print("Successfully added visualization cell to notebook.")
