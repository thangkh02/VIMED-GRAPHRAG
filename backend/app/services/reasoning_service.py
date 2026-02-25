from typing import List, Dict, Any, Set
import networkx as nx
from backend.app.services.graph_service import graph_service

class ReasoningService:
    def __init__(self):
        pass

    @property
    def graph(self):
        return graph_service.graph

    def get_connected_nodes(self, node_name: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """Get connected nodes with confidence filtering (MultiDiGraph compatible)"""
        connected = []
        if self.graph.has_node(node_name):
            # Outgoing edges
            if hasattr(self.graph, "out_edges"):
                edges = self.graph.out_edges(node_name, keys=True, data=True)
            else:
                edges = self.graph.edges(node_name, keys=True, data=True)
                
            for _, neighbor, key, data in edges:
                conf = data.get("confidence", 0)
                if conf >= confidence_threshold:
                    connected.append({
                        "node": neighbor, 
                        "relation": data.get("relation"),
                        "confidence": conf, 
                        "evidence": data.get("evidence", ""), 
                        "key": key
                    })
        return connected

    def explore_path(self, start_node: str, max_depth: int = 2, confidence_threshold: float = 0.5) -> List[Dict]:
        """Find paths from start_node up to max_depth"""
        paths = []
        visited = set()

        def dfs(node, path, accumulated_confidence, depth):
            if depth > max_depth or node in visited: 
                return
            
            visited.add(node)
            
            if len(path) > 0:
                paths.append({
                    'path': path.copy(), 
                    'confidence': accumulated_confidence, 
                    'final_node': node
                })
            
            for neighbor_data in self.get_connected_nodes(node, confidence_threshold):
                neighbor = neighbor_data['node']
                if neighbor in visited: continue
                
                new_confidence = accumulated_confidence * neighbor_data['confidence']
                if new_confidence >= confidence_threshold:
                    new_path = path + [(node, neighbor, neighbor_data['relation'])]
                    dfs(neighbor, new_path, new_confidence, depth + 1)
            
            visited.remove(node)

        dfs(start_node, [], 1.0, 0)
        return paths

    def reason_about_entity(self, entity_name: str, context_depth: int = 2) -> str:
        """Generate reasoning context for a specific entity"""
        if not self.graph.has_node(entity_name): 
            return ""
        
        node_data = self.graph.nodes[entity_name]
        context = f"## Entity: {entity_name}\nType: {node_data.get('type')}\nConfidence: {node_data.get('confidence', 0):.2f}\n"
        if node_data.get('description'):
            context += f"Description: {node_data.get('description')}\n"
            
        context += "\n### Direct Relations:\n"
        connections = self.get_connected_nodes(entity_name, 0.3)
        # Sort by confidence
        connections.sort(key=lambda x: x['confidence'], reverse=True)
        
        for conn in connections[:10]:
            context += f"- {conn['relation']} -> {conn['node']} (conf: {conn['confidence']:.2f})\n"
            
        if context_depth > 1:
            context += "\n### Reasoning Paths (Multi-hop):\n"
            paths = self.explore_path(entity_name, context_depth, 0.3)
            # Sort by confidence
            top_paths = sorted(paths, key=lambda x: x['confidence'], reverse=True)[:5]
            
            for p in top_paths:
                path_str = " -> ".join([f"[{step[2]}] -> {step[1]}" for step in p['path']])
                if path_str: 
                    context += f"- {entity_name} {path_str} (conf: {p['confidence']:.2f})\n"
                    
        return context

reasoning_service = ReasoningService()
