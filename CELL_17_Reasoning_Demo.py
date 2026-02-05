# =========================================================================
# CELL 17: EXAMPLE - REASONING DEMO
# Copy toàn bộ code dưới đây vào Cell 17 của notebook (sau Cell 16)
# =========================================================================

# Demo: Reason about a specific entity
demo_entity = "Bệnh Thận Mạn"  # Change this to any entity in your graph

if demo_entity in G:
    print("🧠 REASONING ABOUT ENTITY\n")
    print(reason_about_entity(G, demo_entity, context_depth=2))
    
    print("\n" + "="*60)
    print("🔗 RELATED ENTITIES\n")
    
    related = find_related_entities(G, demo_entity, top_k=10, min_confidence=0.3)
    
    for r in related:
        print(f"   {r['entity']} ({r['type']}): {r['relevance_score']:.3f}")
    
    print("\n" + "="*60)
    print("📊 EXPLORING PATHS\n")
    
    # Show some paths from this entity
    paths = explore_path(G, demo_entity, max_depth=2, confidence_threshold=0.3)
    top_5_paths = sorted(paths, key=lambda x: x['confidence'], reverse=True)[:5]
    
    for i, path_data in enumerate(top_5_paths, 1):
        path_str = " → ".join([f"{p[0]} [{p[2]}]" for p in path_data['path']])
        if path_str:
            print(f"  {i}. {path_str} → {path_data['final_node']}")
            print(f"     Confidence: {path_data['confidence']:.3f}\n")
else:
    print(f"⚠️ Entity '{demo_entity}' not found in graph.")
    print(f"   Try one of these: {list(G.nodes())[:5]}")
