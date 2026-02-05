# =========================================================================
# CELL 16: GRAPH REASONING & PATH EXPLORATION
# Copy toàn bộ code dưới đây vào Cell 16 của notebook AMG_Improved_Entity_Extraction.ipynb
# =========================================================================

def get_connected_nodes(G, node_name: str, confidence_threshold: float = 0.5):
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


def explore_path(G, start_node: str, max_depth: int = 3, confidence_threshold: float = 0.5):
    """
    Explore paths from a starting node with confidence propagation
    
    Args:
        G: NetworkX DiGraph
        start_node: Starting entity name
        max_depth: Maximum path depth (default 3)
        confidence_threshold: Minimum confidence for paths (default 0.5)
    
    Returns:
        List of paths with confidence scores
    """
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
        
        for neighbor_data in get_connected_nodes(G, node, confidence_threshold):
            neighbor = neighbor_data['node']
            new_confidence = accumulated_confidence * neighbor_data['confidence']
            
            if new_confidence >= confidence_threshold:
                new_path = path + [(node, neighbor, neighbor_data['relation'])]
                dfs(neighbor, new_path, new_confidence, depth + 1)
        
        visited.remove(node)
    
    dfs(start_node, [], 1.0, 0)
    return paths


def reason_about_entity(G, entity_name: str, context_depth: int = 2):
    """
    Generate reasoning context about an entity using graph structure
    
    Args:
        G: NetworkX DiGraph
        entity_name: Entity to reason about
        context_depth: How many hops to explore (default 2)
    
    Returns:
        Reasoning context string
    """
    if entity_name not in G:
        return f"Entity '{entity_name}' not found in graph."
    
    # Get entity info
    node_data = G.nodes[entity_name]
    
    context = f"## Entity: {entity_name}\n"
    context += f"Type: {node_data.get('type', 'Unknown')}\n"
    context += f"Confidence: {node_data.get('confidence', 0):.2f}\n\n"
    
    # Direct connections (depth 1)
    context += "### Direct Connections:\n"
    direct_connections = get_connected_nodes(G, entity_name, confidence_threshold=0.3)
    
    if direct_connections:
        for conn in direct_connections[:5]:  # Top 5
            context += f"- {conn['relation']} → {conn['node']} (conf: {conn['confidence']:.2f})\n"
            if conn['evidence']:
                context += f"  Evidence: {conn['evidence'][:100]}...\n"
    else:
        context += "- No direct connections found\n"
    
    # Explore paths (depth 2+)
    if context_depth > 1:
        context += "\n### Reasoning Paths:\n"
        paths = explore_path(G, entity_name, max_depth=context_depth, confidence_threshold=0.3)
        
        if paths:
            # Show top paths by confidence
            top_paths = sorted(paths, key=lambda x: x['confidence'], reverse=True)[:3]
            
            for path_data in top_paths:
                path_str = " → ".join([f"{p[0]} [{p[2]}]" for p in path_data['path']])
                if path_str:
                    context += f"- {path_str} → {path_data['final_node']} (conf: {path_data['confidence']:.2f})\n"
        else:
            context += "- No multi-hop paths found\n"
    
    return context


def find_related_entities(G, entity_name: str, top_k: int = 10, min_confidence: float = 0.5):
    """
    Find entities most related to a given entity
    
    Args:
        G: NetworkX DiGraph
        entity_name: Source entity
        top_k: Number of related entities to return
        min_confidence: Minimum edge confidence
    
    Returns:
        List of related entities with scores
    """
    related = []
    
    if entity_name not in G:
        return related
    
    # Score entities by: direct connection confidence + number of shared neighbors
    entity_scores = {}
    
    # Direct connections
    for neighbor in G.neighbors(entity_name):
        edge_data = G[entity_name][neighbor]
        conf = edge_data.get('confidence', 0)
        
        if conf >= min_confidence:
            entity_scores[neighbor] = entity_scores.get(neighbor, 0) + conf
    
    # Shared neighbors (2-hop)
    for neighbor in G.neighbors(entity_name):
        for second_hop in G.neighbors(neighbor):
            if second_hop != entity_name and second_hop not in entity_scores:
                # Propagate confidence
                conf1 = G[entity_name][neighbor].get('confidence', 0)
                conf2 = G[neighbor][second_hop].get('confidence', 0)
                propagated_conf = conf1 * conf2 * 0.5  # Decay factor
                
                if propagated_conf >= min_confidence:
                    entity_scores[second_hop] = entity_scores.get(second_hop, 0) + propagated_conf
    
    # Sort and return top_k
    sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    for entity, score in sorted_entities:
        entity_type = G.nodes[entity].get('type', 'Unknown')
        related.append({
            'entity': entity,
            'type': entity_type,
            'relevance_score': score
        })
    
    return related


print("✅ Graph Reasoning functions ready")
print("\nFunctions:")
print("  1. get_connected_nodes(G, node_name, confidence_threshold)")
print("  2. explore_path(G, start_node, max_depth, confidence_threshold)")
print("  3. reason_about_entity(G, entity_name, context_depth)")
print("  4. find_related_entities(G, entity_name, top_k, min_confidence)")
