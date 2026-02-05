# Round 2 Fixes - MultiDiGraph Compatibility

## ✅ Đã Sửa

### 1. **CRITICAL**: Graph Reasoning Multi DiGraph API (Cell 16)

**Vấn đề:**
```python
# ❌ Old (crashes with MultiDiGraph)
for neighbor in G.neighbors(node):
    edge_data = G[node][neighbor]  # Returns dict of edges, not single edge
    conf = edge_data.get('confidence')  # Error!
```

**Giải pháp:**
```python
# ✅ Fixed
for _, neighbor, key, data in G.out_edges(node, keys=True, data=True):
    conf = data.get("confidence", 0)  # Works!
```

**Cells affected**: 14, 16

---

### 2. **CRITICAL**: UNKNOWN Nodes Không Được Upgrade (Cell 8)

**Vấn đề:**
- Placeholder node `type="UNKNOWN"` được tạo khi có relation nhưng thiếu entity
- Khi entity thật xuất hiện sau, chỉ update confidence nhưng KHÔNG update type/label
- → Graph có nhiều nodes "UNKNOWN" vĩnh viễn

**Giải pháp:**
```python
# ✅ Added in add_entity_to_graph()
if G.nodes[norm_name].get("type") == "UNKNOWN":
    G.nodes[norm_name]["type"] = entity.type.upper()
    G.nodes[norm_name]["label"] = entity.name
    G.nodes[norm_name]["description"] = entity.description
```

---

### 3. **HIGH**: Duplicate Edges (Cell 8)

**Vấn đề:**
- MultiDiGraph cho phép nhiều edges
- Nhưng cùng relation + chunk ID lặp lại nhiều lần → phình graph

**Giải pháp:**
```python
def edge_exists(G, src, tgt, rel, chunk_id):
    if not G.has_edge(src, tgt): return False
    edge_dict = G.get_edge_data(src, tgt)
    for key, data in edge_dict.items():
        if data.get("relation") == rel and data.get("chunk") == chunk_id:
            return True
    return False

# Check before adding
if not edge_exists(G, src, tgt, rel_type, chunk_id):
    G.add_edge(...)
```

---

### 4. **MEDIUM**: .title() Phá Medical Abbreviations (Cell 3)

**Vấn đề:**
```python
# ❌ Old
text.title()
# eGFR → Egfr ❌
# HbA1c → Hba1C ❌  
# ACEI → Acei ❌
```

**Giải pháp:**
```python
# ✅ Fixed: Only capitalize first char
if text:
    text = text[0].upper() + text[1:]
# eGFR → EGFR ✅ (via MEDICAL_ABBREVIATIONS)
# HbA1c → HbA1c ✅
```

**Bonus**: Thêm abbreviations vào dict:
```python
MEDICAL_ABBREVIATIONS = {
    "gfr": "GFR", "egfr": "eGFR", 
    "hba1c": "HbA1c", "ldl": "LDL", ...
}
```

---

## 📋 Suggestions (Nice to Have)

### 5. Thêm Entity Types

**Hiện tại**: 9 types
**Đề xuất thêm**:
- `DEVICE` (máy thở, catheter, stent)
- `DOSAGE` (liều lượng, tần suất)
- `PATHOGEN` (vi khuẩn, virus)
- `PATIENT_GROUP` (thai phụ, người cao tuổi)

→ Chưa implement, có thể thêm sau nếu cần

---

### 6. DIAGNOSES vs INDICATES Chồng Lấp

**Vấn đề**: 
- `eGFR DIAGNOSES Bệnh thận mạn`
- `eGFR INDICATES Bệnh thận mạn`
- LLM có thể lẫn lộn

**Đề xuất**:
- `DIAGNOSES`: TEST → DISEASE
- `INDICATES`: LAB_VALUE → DRUG/TREATMENT

→ Chưa thay đổi, giữ nguyên 12 types

---

### 7. Checkpoint Fingerprint

**Đề xuất**: Lưu hash(PDF + chunk_size + model) để detect config changes

→ Đã tạo code trong `ROUND2_FIXES.py` nhưng chưa apply vào notebook  
→ Có thể thêm sau nếu cần

---

## 📊 Summary

| Issue | Mức Độ | Cell | Status |
|-------|---------|------|--------|
| MultiDiGraph API incompatible | CRITICAL | 14, 16 | ✅ FIXED |
| UNKNOWN upgrade missing | CRITICAL | 8 | ✅ FIXED |
| Duplicate edges | HIGH | 8 | ✅ FIXED |
| .title() breaks abbreviations | MEDIUM | 3 | ✅ FIXED |
| More entity types | NICE TO HAVE | - | ⏭️ LATER |
| Relation overlap | NICE TO HAVE | - | ⏭️ LATER |
| Checkpoint fingerprint | NICE TO HAVE | 9 | ⏭️ LATER |

---

## 🎯 Testing

```python
# Test 1: Reasoning with MultiDiGraph
demo_entity = "Bệnh Thận Mạn"
print(reason_about_entity(G, demo_entity))  # Should work!

# Test 2: UNKNOWN upgrade
# 1. Add relation (creates UNKNOWN placeholders)
# 2. Add entities later
# 3. Check node types - should be upgraded from UNKNOWN

# Test 3: No duplicates
# Add same relation twice → Check G.number_of_edges() unchanged

# Test 4: Abbreviations preserved
print(normalize_text("egfr < 60"))  # Should be "EGFR < 60" not "Egfr < 60"
```

---

**⚠️ LƯU Ý**: Nếu có checkpoint cũ, XÓA và chạy lại từ đầu!
