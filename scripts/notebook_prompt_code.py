# ==========================================
# CHO VÀO NOTEBOOK: AMG_RAG_Final_Improved.ipynb
# ==========================================
# Copy đoạn code này vào một cell mới trong notebook của bạn

# Option 1: Import trực tiếp từ amg_utils.py (KHUYẾN NGHỊ)
# ----------------------------------------------------
from langchain_core.prompts import ChatPromptTemplate

# Import prompt đã cải thiện từ amg_utils.py
import sys
sys.path.append('.')  # Đảm bảo thư mục hiện tại trong path

from amg_utils import get_entity_extraction_prompt

# Sử dụng prompt
extraction_prompt = get_entity_extraction_prompt()
print("✅ Đã load prompt cải thiện từ amg_utils.py")
print(f"   - 18 relation types")
print(f"   - 4 few-shot examples")
print(f"   - Hướng dẫn chi tiết")


# Option 2: Nếu muốn code trực tiếp trong notebook (không khuyến nghị)
# --------------------------------------------------------------------
# Copy toàn bộ đoạn code dưới đây:

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

## FEW-SHOT EXAMPLES:

### Example 1:
Input: "Bệnh thận mạn giai đoạn 5 cần lọc máu. Chỉ định dùng Erythropoietin khi Hb < 10g/dL."

Entities:
- Bệnh thận mạn giai đoạn 5 (DISEASE, score=10, desc="Mức độ nặng nhất của bệnh thận mạn")
- Lọc máu (TREATMENT, score=9, desc="Phương pháp thay thế thận")
- Erythropoietin (DRUG, score=8, desc="Thuốc điều trị thiếu máu")
- Hb < 10g/dL (LAB_VALUE, score=7, desc="Ngưỡng chỉ định điều trị")

Relations:
- Lọc máu TREATS Bệnh thận mạn giai đoạn 5 (confidence=9, evidence="cần lọc máu")
- Hb < 10g/dL INDICATES Erythropoietin (confidence=8, evidence="Chỉ định dùng Erythropoietin khi Hb < 10g/dL")

### Example 2:
Input: "Tiểu đường type 2 là yếu tố nguy cơ hàng đầu gây bệnh thận mạn. Kiểm soát đường huyết bằng Metformin."

Entities:
- Đái tháo đường type 2 (DISEASE, score=9, desc="Nguyên nhân chính gây BTM")
- Bệnh thận mạn (DISEASE, score=9, desc="Biến chứng của tiểu đường")
- Metformin (DRUG, score=8, desc="Thuốc kiểm soát đường huyết")
- Đường huyết (TEST, score=6, desc="Chỉ số cần kiểm soát")

Relations:
- Đái tháo đường type 2 CAUSES Bệnh thận mạn (confidence=9, evidence="yếu tố nguy cơ hàng đầu gây bệnh thận mạn")
- Đái tháo đường type 2 INCREASES_RISK Bệnh thận mạn (confidence=10, evidence="yếu tố nguy cơ hàng đầu")
- Metformin TREATS Đái tháo đường type 2 (confidence=8, evidence="Kiểm soát đường huyết bằng Metformin")

### Example 3:
Input: "ACE inhibitor làm giảm protein niệu ở bệnh nhân đái tháo đường, bảo vệ chức năng thận."

Entities:
- ACE inhibitor (DRUG, score=9, desc="Thuốc ức chế men chuyển")
- Protein niệu (SYMPTOM, score=8, desc="Tình trạng có protein trong nước tiểu")
- Đái tháo đường (DISEASE, score=7, desc="Bệnh nền gây protein niệu")
- Thận (ANATOMY, score=6, desc="Cơ quan được bảo vệ")

Relations:
- ACE inhibitor TREATS Protein niệu (confidence=9, evidence="làm giảm protein niệu")
- ACE inhibitor PREVENTS Suy giảm chức năng thận (confidence=8, evidence="bảo vệ chức năng thận")
- Đái tháo đường CAUSES Protein niệu (confidence=7, evidence="protein niệu ở bệnh nhân đái tháo đường")

### Example 4 (Sử dụng relation types mới):
Input: "Buồn nôn và rụng tóc là tác dụng phụ thường gặp của hóa trị. Thuốc chống nôn giúp giảm triệu chứng."

Entities:
- Buồn nôn (SYMPTOM, score=8, desc="Triệu chứng phổ biến")
- Rụng tóc (SYMPTOM, score=7, desc="Tác dụng phụ thường gặp")
- Hóa trị (TREATMENT, score=9, desc="Điều trị ung thư")
- Thuốc chống nôn (DRUG, score=7, desc="Thuốc giảm buồn nôn")

Relations:
- Buồn nôn SIDE_EFFECT_OF Hóa trị (confidence=9, evidence="tác dụng phụ thường gặp của hóa trị")
- Rụng tóc SIDE_EFFECT_OF Hóa trị (confidence=9, evidence="tác dụng phụ thường gặp của hóa trị")
- Thuốc chống nôn ALLEVIATES Buồn nôn (confidence=8, evidence="giúp giảm triệu chứng")

## HƯỚNG DẪN CHỌN RELATION TYPE:
1. **Ưu tiên Core Relations** (CAUSES, TREATS, PREVENTS, DIAGNOSES) khi có thể
2. **Sử dụng specific relations** (SYMPTOM_OF, COMPLICATION_OF, SIDE_EFFECT_OF) khi mối quan hệ rõ ràng
3. **Dùng RELATED_TO** CHÍNH khi:
   - Mối quan hệ không khớp với bất kỳ type nào trên
   - Quan hệ phức tạp, cần mô tả chi tiết trong evidence
4. **Tránh trùng lặp**: Nếu có nhiều relation type phù hợp, chọn type CỤ THỂ nhất
   - VD: Ưu tiên COMPLICATION_OF hơn RELATED_TO
   - VD: Ưu tiên SIDE_EFFECT_OF hơn CAUSES

## RULES (Quy tắc):
1. **GIỮ NGUYÊN** tên tiếng Việt, KHÔNG dịch sang tiếng Anh
2. **KHÔNG trích xuất**: số trang, quyết định, văn bản, tên tác giả, tên người
3. **Mô tả entity** phải NGẮN GỌN (1 câu)
4. **Evidence cho relation** phải trích từ văn bản gốc
5. **CHỈ trích xuất relations** khi có bằng chứng rõ ràng trong văn bản
6. **Ưu tiên QUALITY hơn QUANTITY** - chỉ extract những gì chắc chắn
7. **Tránh trùng lặp** - chuẩn hóa tên entities (VD: "BTM" → "Bệnh thận mạn")

## NEGATIVE EXAMPLES (KHÔNG trích xuất):
- "Bộ Y tế", "Quyết định số 123", "Trang 45" → Nội dung hành chính
- "Theo nghiên cứu năm 2020" → Thông tin meta
- "Bệnh nhân Nguyễn Văn A" → Tên người cụ thể

Bây giờ, hãy trích xuất ENTITIES và RELATIONS từ văn bản sau:"""),
    ("human", "{text}")
])

print("✅ Đã định nghĩa prompt cải thiện trực tiếp trong notebook")
