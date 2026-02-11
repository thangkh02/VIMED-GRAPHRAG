import networkx as nx
from typing import List, Dict, Any, Optional

class ReasoningService:
    def __init__(self):
        # In a real app, we might load a persistent graph here or pass it in
        pass

    def get_connected_nodes(self, G: nx.DiGraph, node_name: str, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Get nodes connected to a given node with confidence above threshold"""
        connected = []
        if node_name in G:
            for neighbor in G.neighbors(node_name):
                edge_data = G[node_name][neighbor]
                if edge_data.get('confidence', 0) >= confidence_threshold:
                    connected.append({
                        'node': neighbor,
                        'relation': edge_data.get('relation'),
                        'confidence': edge_data.get('confidence'),
                        'evidence': edge_data.get('evidence')
                    })
        return connected

    def explore_path(self, G: nx.DiGraph, start_node: str, max_depth: int = 3, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Explore paths from a starting node with confidence propagation"""
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
            
            for neighbor_data in self.get_connected_nodes(G, node, confidence_threshold):
                neighbor = neighbor_data['node']
                new_confidence = accumulated_confidence * neighbor_data['confidence']
                
                if new_confidence >= confidence_threshold:
                    new_path = path + [(node, neighbor, neighbor_data['relation'])]
                    dfs(neighbor, new_path, new_confidence, depth + 1)
            
            visited.remove(node)
        
        dfs(start_node, [], 1.0, 0)
        return paths

    def find_shortest_path(self, G: nx.DiGraph, source: str, target: str, max_length: int = 5) -> Optional[List[Dict[str, Any]]]:
        """Find shortest path between two entities"""
        try:
            path = nx.shortest_path(G, source=source, target=target)
            
            if len(path) > max_length + 1:
                return None
            
            path_with_relations = []
            for i in range(len(path) - 1):
                edge_data = G[path[i]][path[i+1]]
                path_with_relations.append({
                    'from': path[i],
                    'to': path[i+1],
                    'relation': edge_data.get('relation'),
                    'confidence': edge_data.get('confidence'),
                    'evidence': edge_data.get('evidence', '')[:100]
                })
            
            return path_with_relations
        except nx.NetworkXNoPath:
            return None
        except Exception:
            return None

    def reason_about_entity(self, G: nx.DiGraph, entity_name: str, context_depth: int = 2) -> str:
        """Generate reasoning context about an entity using graph structure"""
        if entity_name not in G:
            return f"Entity '{entity_name}' not found in graph."
        
        node_data = G.nodes[entity_name]
        
        context = f"## Entity: {entity_name}\n"
        context += f"Type: {node_data.get('type', 'Unknown')}\n"
        context += f"Description: {node_data.get('description', 'No description')}\n"
        context += f"Confidence: {node_data.get('confidence', 0):.2f}\n\n"
        
        # Direct connections
        context += "### Direct Connections:\n"
        direct_connections = self.get_connected_nodes(G, entity_name, confidence_threshold=0.3)
        
        if direct_connections:
            for conn in direct_connections[:5]:
                context += f"- {conn['relation']} → {conn['node']} (conf: {conn['confidence']:.2f})\n"
                if conn['evidence']:
                    context += f"  Evidence: {conn['evidence'][:100]}...\n"
        else:
            context += "- No direct connections found\n"
        
        # Explore paths
        if context_depth > 1:
            context += "\n### Reasoning Paths:\n"
            paths = self.explore_path(G, entity_name, max_depth=context_depth, confidence_threshold=0.3)
            
            if paths:
                top_paths = sorted(paths, key=lambda x: x['confidence'], reverse=True)[:3]
                for path_data in top_paths:
                    path_str = " → ".join([f"{p[0]} [{p[2]}]" for p in path_data['path']])
                    if path_str:
                        context += f"- {path_str} → {path_data['final_node']} (conf: {path_data['confidence']:.2f})\n"
            else:
                context += "- No multi-hop paths found\n"
        
        return context

reasoning_service = ReasoningService()
