# Critical Fixes Applied to AMG_Improved_Entity_Extraction.ipynb

## 🐛 Vấn Đề Phát Hiện & Giải Pháp

### **FIX 1: Inverse Relations Không Nằm Trong Whitelist** ⚠️ CRITICAL

**Vấn đề:**
- `generate_inverse_relations()` tạo ra relation types mới (`CAUSED_BY`, `TREATED_BY`, etc.)
- Các types này KHÔNG nằm trong `VALID_RELATION_TYPES` (chỉ có 12 types gốc)
- Khi query/reasoning, các relation "lạ" này gây confusion

**Hậu quả:**
- Graph chứa relation types không theo schema
- Thống kê relation types sai
- Reasoning logic bị rối khi gặp types không nhận diện

**Giải pháp:**
✅ **Thêm 10 inverse types vào `VALID_RELATION_TYPES`:**

```python
VALID_RELATION_TYPES = {
    # Original 12 types
    "CAUSES", "TREATS", "PREVENTS", "DIAGNOSES",
    "SYMPTOM_OF", "COMPLICATION_OF", "SIDE_EFFECT_OF", "INCREASES_RISK",
    "INTERACTS_WITH", "WORSENS", "INDICATES",
    "RELATED_TO",
    
    # ✅ NEW: 10 inverse types
    "CAUSED_BY", "TREATED_BY", "PREVENTED_BY", "DIAGNOSED_BY",
    "HAS_SYMPTOM", "HAS_COMPLICATION", "HAS_SIDE_EFFECT", 
    "RISK_INCREASED_BY", "WORSENED_BY", "INDICATED_BY"
}
```

**Lý do:** Giữ schema nhất quán, validation pass cho cả original và inverse relations.

---

### **FIX 2: Inverse Confidence Có Thể = 5 (Dưới Threshold)** ⚠️ CRITICAL

**Vấn đề:**
```python
# Code cũ
inverse_confidence = max(5, rel.confidence_score - 1)
```
- Nếu `rel.confidence_score = 6` → inverse = `5`
- Nhưng validation rule: `confidence < 6` thì reject
- Tạo mâu thuẫn: tạo ra relation rồi lại reject

**Hậu quả:**
- Inverse relations yếu (conf = 5) vẫn lọt vào graph
- Tăng noise, giảm chất lượng graph
- Không nhất quán với validation rule

**Giải pháp:**
✅ **Đảm bảo inverse confidence >= 6:**

```python
# ✅ Fixed
inverse_confidence = max(6, rel.confidence_score - 1)
```

**Lý do:** Tuân thủ strict rule `confidence >= 6` cho TẤT CẢ relations.

---

### **FIX 3: DiGraph Mất Multi-Relations** ⚠️ CRITICAL

**Vấn đề:**
- `nx.DiGraph()` chỉ giữ **1 edge** cho mỗi cặp `(src, tgt)`
- Nếu 2 entities có nhiều quan hệ (VD: `A CAUSES B`, `A WORSENS B`)
- → Chỉ giữ lại relation cuối cùng, mất relation trước

**Hậu quả:**
- Mất thông tin quan trọng (y tế thường có nhiều quan hệ giữa 2 entities)
- Mất evidence từ nhiều chunks khác nhau
- Graph "nghèo thông tin"

**Ví dụ:**
```python
# Chunk 50: "Tiểu đường gây bệnh thận mạn"
A CAUSES B (conf=9, evidence="gây")

# Chunk 100: "Tiểu đường làm nặng bệnh thận mạn"
A WORSENS B (conf=8, evidence="làm nặng")

# DiGraph → Chỉ giữ lại edge cuối (WORSENS), mất CAUSES ❌
```

**Giải pháp:**
✅ **Chuyển sang `nx.MultiDiGraph()`:**

```python
# Cell 10
G = nx.MultiDiGraph()  # Cho phép nhiều edges giữa 2 nodes
```

✅ **Tạo placeholder nodes để không mất relations:**

```python
def add_relation_to_graph(G, rel, page_num, chunk_id):
    src = normalize_text(rel.source_name)
    tgt = normalize_text(rel.target_name)
    
    # ✅ Tạo placeholder nếu entity chưa tồn tại
    if not G.has_node(src):
        G.add_node(src, label=rel.source_name, type="UNKNOWN", confidence=0.5, ...)
    if not G.has_node(tgt):
        G.add_node(tgt, label=rel.target_name, type="UNKNOWN", confidence=0.5, ...)
    
    # ✅ MultiDiGraph: Add edge (không overwrite)
    G.add_edge(src, tgt, relation=..., confidence=..., evidence=...)
```

**Lý do:** 
- Y tế có nhiều quan hệ phức tạp giữa 2 entities
- Giữ đầy đủ evidence từ nhiều chunks
- Không mất thông tin khi merge

---

### **FIX 4: Skip Chunk Khi Entities Rỗng (Dù Có Relations)** ⚠️ HIGH

**Vấn đề:**
```python
# Code cũ
if not result or not result.entities:
    continue  # ❌ Bỏ qua toàn bộ chunk
```
- LLM đôi khi parse fail entities nhưng vẫn trả relations
- Hoặc chunk chỉ có relations, không có entity mới
- → Mất toàn bộ relations của chunk đó

**Hậu quả:**
- Recall tụt mạnh (mất nhiều relations)
- Với tài liệu dài, mất đáng kể information

**Giải pháp:**
✅ **Chỉ skip khi CẢ 2 đều rỗng:**

```python
# ✅ Fixed
if not result or (not result.entities and not result.relations):
    continue
```

**Lý do:** Giữ lại relations ngay cả khi entities rỗng (placeholder nodes sẽ được tạo).

---

### **FIX 5: Checkpoint Cuối Ghi Sai** ⚠️ CRITICAL

**Vấn đề:**
```python
# Code cũ
checkpoint_manager.save(G, len(chunks)-1, len(chunks))
# ❌ Luôn ghi last_chunk_id = len(chunks)-1
```
- Dù vòng lặp chưa chắc chạy hết (error, stop, skip chunks)
- Lần sau resume → nghĩ "đã xong" → không xử lý phần còn lại

**Hậu quả:**
- Lỗi cực khó phát hiện (nhìn checkpoint tưởng đúng)
- Mất data thầm lặng

**Giải pháp:**
✅ **Track actual last processed chunk:**

```python
# ✅ Fixed
last_processed_chunk = start_chunk - 1  # Init

for i, chunk in enumerate(chunks_to_process, start=start_chunk):
    # ... process ...
    last_processed_chunk = i  # Update after each chunk
    
    if (i + 1) % 20 == 0:
        checkpoint_manager.save(G, i, len(chunks))

# Final: Save ACTUAL last processed chunk
checkpoint_manager.save(G, last_processed_chunk, len(chunks))
```

**Lý do:** Đảm bảo checkpoint chính xác, resume đúng vị trí.

---

## 📊 Tổng Kết

| Fix | Vấn Đề | Mức Độ | Cell Affected |
|-----|--------|--------|---------------|
| 1 | Inverse types không trong whitelist | CRITICAL | Cell 6 |
| 2 | Inverse confidence < 6 | CRITICAL | Cell 6 |
| 3 | DiGraph mất multi-relations | CRITICAL | Cell 8, 9, 10, 13 |
| 4 | Skip chunk khi entities rỗng | HIGH | Cell 12 |
| 5 | Checkpoint cuối sai | CRITICAL | Cell 12 |

---

## ✅ Cells Đã Fix

- **Cell 6**: Validation & Inverse Relations
- **Cell 8**: Graph Helpers (MultiDiGraph + placeholders)
- **Cell 9**: Checkpoint Manager (type hints)
- **Cell 10**: Graph Creation (MultiDiGraph)
- **Cell 12**: Main Loop (skip logic + tracking)
- **Cell 13**: Statistics (MultiDiGraph iteration)

---

## 🎯 Testing Checklist

- [ ] Kiểm tra `VALID_RELATION_TYPES` có 22 types (12 original + 10 inverse)
- [ ] Verify inverse relations có confidence >= 6
- [ ] Test nhiều relations giữa 2 nodes (ví dụ: A CAUSES B và A WORSENS B)
- [ ] Test chunk có relations nhưng không có entities
- [ ] Test resume từ checkpoint (kiểm tra last_chunk_id chính xác)
- [ ] Chạy Cell 13 statistics để xem relation distribution

---

**⚠️ IMPORTANT:** Nếu bạn đã có checkpoint cũ (DiGraph), cần xóa và chạy lại từ đầu với MultiDiGraph!
