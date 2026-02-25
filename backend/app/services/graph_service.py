import networkx as nx
import os
import pickle
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from backend.app.core.config import settings
from backend.app.models.schemas import Entity, Relation
from backend.app.services.text_processing import normalize_medical_text
from pyvis.network import Network

class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.graph_path = os.path.join(checkpoint_dir, "graph_improved.pkl")
        self.meta_path = os.path.join(checkpoint_dir, "checkpoint_meta.json")
    
    def save(self, G: nx.MultiDiGraph, chunk_id: int, total_chunks: int):
        """Save graph and metadata"""
        with open(self.graph_path, "wb") as f:
            pickle.dump(G, f)
        
        meta = {
            "last_chunk_id": chunk_id, 
            "total_chunks": total_chunks, 
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(), 
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"Checkpoint saved: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    def load(self) -> tuple[Optional[nx.MultiDiGraph], Optional[int]]:
        """Load graph and last chunk_id"""
        graph, last_chunk_id = None, None
        if os.path.exists(self.graph_path):
            with open(self.graph_path, "rb") as f:
                graph = pickle.load(f)
            print(f"Loaded graph: {graph.number_of_nodes()} nodes")
        
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            last_chunk_id = meta.get("last_chunk_id")
        
        return graph, last_chunk_id

class GraphService:
    def __init__(self):
        self.checkpoint_manager = CheckpointManager(os.path.join(settings.DATA_DIR, "amg_data"))
        self.graph, self.last_chunk_id = self.checkpoint_manager.load()
        if self.graph is None:
            self.graph = nx.MultiDiGraph()
            print("Initialized new MultiDiGraph")

    def save_checkpoint(self, chunk_id: int, total_chunks: int):
        self.checkpoint_manager.save(self.graph, chunk_id, total_chunks)

    def add_entity(self, entity: Entity, page_num: int, chunk_id: int):
        """Add or update entity in the graph"""
        norm_name = normalize_medical_text(entity.name)
        confidence = min(1.0, entity.relevance_score / 10.0)
        
        if not self.graph.has_node(norm_name):
            self.graph.add_node(
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
            # Upgrade from UNKNOWN if possible
            node = self.graph.nodes[norm_name]
            if node.get("type") == "UNKNOWN" and entity.type.upper() != "UNKNOWN":
                node["type"] = entity.type.upper()
                node["label"] = entity.name
                node["description"] = entity.description
            
            # Update confidence if higher
            old_conf = node.get('confidence', 0)
            if confidence > old_conf:
                node['confidence'] = confidence
                if node.get("type") != "UNKNOWN":
                    node['description'] = entity.description
            
            # Track pages/chunks
            if page_num not in node['pages']:
                node['pages'].append(page_num)
            if chunk_id not in node['chunks']:
                node['chunks'].append(chunk_id)

    def edge_exists(self, src, tgt, rel_type, chunk_id):
        if not self.graph.has_edge(src, tgt): return False
        edge_data = self.graph.get_edge_data(src, tgt)
        for key, data in edge_data.items():
            if data.get("relation") == rel_type and data.get("chunk") == chunk_id:
                return True
        return False

    def add_relation(self, relation: Relation, page_num: int, chunk_id: int):
        """Add relation to the graph with deduplication"""
        src = normalize_medical_text(relation.source_name)
        tgt = normalize_medical_text(relation.target_name)
        rel_type = relation.relation.upper()
        
        # Ensure nodes exist (create as UNKNOWN if missing)
        if not self.graph.has_node(src):
            self.graph.add_node(src, label=relation.source_name, type="UNKNOWN", confidence=0.5, pages=[page_num], chunks=[chunk_id], description="")
        if not self.graph.has_node(tgt):
            self.graph.add_node(tgt, label=relation.target_name, type="UNKNOWN", confidence=0.5, pages=[page_num], chunks=[chunk_id], description="")
            
        # Add edge if not duplicate for this chunk
        if not self.edge_exists(src, tgt, rel_type, chunk_id):
            self.graph.add_edge(
                src, tgt, 
                relation=rel_type, 
                confidence=min(1.0, relation.confidence_score/10.0),
                evidence=relation.evidence, 
                page=page_num, 
                chunk=chunk_id
            )

    def visualize_graph(self, output_filename: str = "current_graph.html") -> str:
        """Generate HTML visualization"""
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", select_menu=True, filter_menu=True)
        net.force_atlas_2based()
        
        color_map = {
            "DISEASE": "#FF6B6B", "DRUG": "#4ECDC4", "SYMPTOM": "#FFE66D",
            "TEST": "#1A535C", "ANATOMY": "#FF9F1C", "TREATMENT": "#2B2D42",
            "PROCEDURE": "#8D99AE", "RISK_FACTOR": "#EF233C", "LAB_VALUE": "#219EBC",
            "UNKNOWN": "#cccccc"
        }
        
        # Add nodes
        for node, data in self.graph.nodes(data=True):
            node_type = data.get("type", "UNKNOWN")
            color = color_map.get(node_type, "#999999")
            degree = self.graph.degree(node)
            size = 15 + (degree * 2)
            title_html = f"<b>{node}</b><br>Type: {node_type}<br>Connections: {degree}"
            if data.get("description"):
                title_html += f"<br><i>{data['description'][:100]}...</i>"
            
            net.add_node(node, label=node, title=title_html, color=color, value=size, group=node_type)
        
        # Add edges (simplify MultiDiGraph for viz by taking best edge or all)
        # Visualizing all edges might be cluttered, but let's do it for now
        for u, v, data in self.graph.edges(data=True):
            rel_type = data.get("relation", "RELATED")
            confidence = data.get("confidence", 0.5)
            net.add_edge(u, v, title=f"{rel_type} (conf: {confidence:.2f})", label=rel_type, width=confidence*3, color="#aaaaaa")
            
        output_path = os.path.join(settings.DATA_DIR, output_filename)
        net.save_graph(output_path)
        return output_path

graph_service = GraphService()
