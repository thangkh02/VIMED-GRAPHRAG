from pyvis.network import Network
import networkx as nx
import os
from backend.app.core.config import settings

class GraphService:
    def __init__(self):
        pass

    def visualize_graph(self, G: nx.DiGraph, output_filename: str = "graph_viz.html") -> str:
        """Generate HTML visualization for the graph"""
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", select_menu=True, filter_menu=True)
        # Force Atlas 2 Based is good for large networks
        net.force_atlas_2based()
        
        # Define intuitive colors for medical entities
        color_map = {
            "DISEASE": "#FF6B6B",       # Red/Salmon
            "DRUG": "#4ECDC4",          # Teal/Green
            "SYMPTOM": "#FFE66D",       # Yellow
            "TEST": "#1A535C",          # Dark Cyan
            "ANATOMY": "#FF9F1C",       # Orange
            "TREATMENT": "#2B2D42",     # Dark Blue
            "PROCEDURE": "#8D99AE",     # Grey Blue
            "RISK_FACTOR": "#EF233C",   # Red
            "LAB_VALUE": "#219EBC",     # Blue
            "UNKNOWN": "#cccccc"
        }
        
        # Add nodes with custom visual properties
        for node, data in G.nodes(data=True):
            node_type = data.get("type", "UNKNOWN")
            color = color_map.get(node_type, "#999999")
            
            # Size based on degree (importance)
            degree = G.degree(node)
            size = 15 + (degree * 2)
            
            title_html = f"<b>{node}</b><br>Type: {node_type}<br>Connections: {degree}"
            if "description" in data and data["description"]:
                title_html += f"<br><i>{data['description'][:100]}...</i>"

            net.add_node(node, label=node, title=title_html, color=color, value=size, group=node_type)
        
        # Add edges
        for u, v, data in G.edges(data=True):
            rel_type = data.get("relation", "RELATED_TO")
            confidence = data.get("confidence", 0.5)
            width = confidence * 3  # Thicker lines for higher confidence
            
            net.add_edge(u, v, title=f"{rel_type} (conf: {confidence:.2f})", label=rel_type, width=width, color="#aaaaaa")
        
        # Save logic
        output_path = os.path.join(settings.DATA_DIR, output_filename)
        net.save_graph(output_path)
        return output_path

graph_service = GraphService()
