from langchain_core.prompts import ChatPromptTemplate

# ==============================================================================
# PROMPTS
# ==============================================================================

ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Bạn là chuyên gia trích xuất thực thể và mối quan hệ y tế cho Medical Knowledge Graph.

## ENTITY TYPES (Loại thực thể):
1. **DISEASE** - Bệnh, hội chứng, rối loạn (VD: Bệnh thận mạn, Đái tháo đường type 2)
2. **DRUG** - Thuốc, hoạt chất, nhóm thuốc (VD: Metformin, ACEI, Statin)
3. **SYMPTOM** - Triệu chứng, dấu hiệu (VD: Phù, Tiểu ít, Mệt mỏi)
4. **TEST** - Xét nghiệm, chẩn đoán (VD: Creatinine, eGFR, Siêu âm thận)
5. **ANATOMY** - Cơ quan, bộ phận cơ thể (VD: Thận, Tim, Mạch máu)
6. **TREATMENT** - Phương pháp điều trị (VD: Lọc máu, Ghép thận, Chế độ ăn)
7. **PROCEDURE** - Thủ thuật, can thiệp (VD: Sinh thiết thận, Đặt catheter)
8. **RISK_FACTOR** - Yếu tố nguy cơ (VD: Hút thuốc, Béo phì, Tiền sử gia đình)
9. **LAB_VALUE** - Giá trị xét nghiệm quan trọng (VD: GFR < 60, Creatinine > 1.5)

## RELATION TYPES (Loại quan hệ):
### Core Relations (Quan hệ cốt lõi - Ưu tiên sử dụng):
- **CAUSES** - Gây ra (VD: Tiểu đường CAUSES Bệnh thận mạn)
- **TREATS** - Điều trị (VD: Insulin TREATS Tiểu đường)
- **PREVENTS** - Phòng ngừa (VD: ACEI PREVENTS Tiến triển bệnh thận)
- **DIAGNOSES** - Chẩn đoán (VD: eGFR DIAGNOSES Bệnh thận mạn)
- **INDICATES** - Chỉ định (VD: Hb < 10 INDICATES Erythropoietin)
- **CONTRAINDICATES** - Chống chỉ định (VD: Metformin CONTRAINDICATES Suy thận nặng)

### Risk Relations (Quan hệ nguy cơ):
- **INCREASES_RISK** - Tăng nguy cơ (VD: Hút thuốc INCREASES_RISK Ung thư)
- **REDUCES_RISK** - Giảm nguy cơ (VD: Tập thể dục REDUCES_RISK Tiểu đường)

### Clinical Relations (Quan hệ lâm sàng):
- **SYMPTOM_OF** - Triệu chứng của (VD: Phù SYMPTOM_OF Suy tim)
- **COMPLICATION_OF** - Biến chứng của (VD: Bệnh võng mạc COMPLICATION_OF Tiểu đường)
- **SIDE_EFFECT_OF** - Tác dụng phụ của (VD: Buồn nôn SIDE_EFFECT_OF Hóa trị)

### Pharmacological Relations (Quan hệ dược lý):
- **INTERACTS_WITH** - Tương tác với (VD: Warfarin INTERACTS_WITH Aspirin)
- **METABOLIZED_BY** - Chuyển hóa bởi (VD: Codeine METABOLIZED_BY CYP2D6)

### Severity Relations (Quan hệ mức độ):
- **WORSENS** - Làm nặng thêm (VD: Mất nước WORSENS Suy thận)
- **ALLEVIATES** - Làm giảm nhẹ (VD: Morphine ALLEVIATES Đau)

### Structural Relations (Quan hệ cấu trúc):
- **LOCATED_IN** - Vị trí tại (VD: Viêm LOCATED_IN Thận)
- **AFFECTS** - Ảnh hưởng (VD: Thuốc lợi tiểu AFFECTS Điện giải)

### Catch-all (Dùng khi không khớp các loại trên):
- **RELATED_TO** - Liên quan đến (ghi rõ chi tiết trong evidence)

## ENTITY SCORING (relevance_score 1-10):
- **9-10**: Thực thể CHÍNH, được đề cập nhiều lần
- **7-8**: Thực thể QUAN TRỌNG, liên quan trực tiếp
- **5-6**: Thực thể PHỤ, được đề cập nhưng không trọng tâm
- **3-4**: Thực thể LIÊN QUAN nhẹ
- **1-2**: Thực thể RẤT PHỤ

## RELATION SCORING (confidence_score 1-10):
- **9-10**: Mối quan hệ RÕ RÀNG, được nêu trực tiếp trong văn bản
- **7-8**: Mối quan hệ MẠNH, có bằng chứng rõ ràng
- **5-6**: Mối quan hệ VỪA PHẢI, được ngụ ý từ ngữ cảnh
- **3-4**: Mối quan hệ YẾU, liên kết gián tiếp
- **1-2**: Mối quan hệ RẤT YẾU, chỉ là suy đoán

## RULES (Quy tắc):
1. **GIỮ NGUYÊN** tên tiếng Việt, KHÔNG dịch sang tiếng Anh
2. **KHÔNG trích xuất**: số trang, quyết định, văn bản, tên tác giả, tên người
3. **Mô tả entity** phải NGẮN GỌN (1 câu)
4. **Evidence cho relation** phải trích từ văn bản gốc
5. **CHỈ trích xuất relations** khi có bằng chứng rõ ràng trong văn bản
6. **Ưu tiên QUALITY hơn QUANTITY** - chỉ extract những gì chắc chắn
7. **Tránh trùng lặp** - chuẩn hóa tên entities (VD: "BTM" → "Bệnh thận mạn")

Bây giờ, hãy trích xuất ENTITIES và RELATIONS từ văn bản sau:"""),
    ("human", "{text}")
])

RELATION_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Phân tích quan hệ Y TẾ 2 CHIỀU giữa 2 thực thể.

## RELATION TYPES:
1. **CAUSES** - A gây ra B (VD: Tiểu đường → CAUSES → Bệnh thận mạn)
2. **TREATS** - A điều trị B (VD: Insulin → TREATS → Tiểu đường)
3. **SYMPTOM_OF** - A là triệu chứng của B (VD: Phù → SYMPTOM_OF → Suy thận)
4. **RISK_FACTOR** - A là yếu tố nguy cơ của B (VD: Hút thuốc → RISK_FACTOR → Ung thư phổi)
5. **DIAGNOSED_BY** - A được chẩn đoán bằng B (VD: Bệnh thận mạn → DIAGNOSED_BY → eGFR)
6. **PREVENTS** - A ngăn ngừa B (VD: ACEI → PREVENTS → Tiến triển bệnh thận)
7. **CONTRAINDICATED** - A chống chỉ định với B (VD: Metformin → CONTRAINDICATED → Suy thận nặng)
8. **ASSOCIATED_WITH** - A liên quan đến B (VD: Tăng huyết áp → ASSOCIATED_WITH → Bệnh tim)
9. **COMPLICATION_OF** - A là biến chứng của B (VD: Bệnh võng mạc → COMPLICATION_OF → Tiểu đường)

## SCORING RUBRIC:
- **9-10**: Quan hệ RẤT CHẮC CHẮN, được ghi rõ trong văn bản
- **7-8**: Quan hệ KHẢ NĂNG CAO, suy luận từ ngữ cảnh
- **5-6**: Quan hệ CÓ THỂ, cần thêm bằng chứng
- **1-4**: Quan hệ YẾU hoặc không chắc chắn

## QUY TẮC:
1. PHẢI có cả quan hệ Forward VÀ Backward
2. Confidence score phải có căn cứ từ context
3. Evidence phải trích dẫn từ văn bản gốc"""),
    ("human", "Entity1: {entity1}\nEntity2: {entity2}\nContext: {context}")
])

ENTITY_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tạo TÓM TẮT Y TẾ cho các thực thể.

## YÊU CẦU:
1. Mỗi entity có tóm tắt 2-3 câu DỰA TRÊN CONTEXT
2. Tập trung vào: định nghĩa, vai trò, tầm quan trọng trong y tế
3. Chấm điểm importance dựa trên mức độ quan trọng trong context

## LƯU Ý:
- KHÔNG thêm thông tin ngoài context
- KHÔNG dài quá 3 câu
- Ưu tiên thông tin LÂM SÀNG hơn là học thuật"""),
    ("human", "Entities: {entities}\nContext: {context}")
])
