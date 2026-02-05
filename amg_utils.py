"""
AMG Utils - Optimized Entity Extraction & Normalization for Medical Domain
"""

import re
import unicodedata
from typing import List, Dict, Tuple
from langchain_core.prompts import ChatPromptTemplate

# ==============================================================================
# PHẦN 1: NORMALIZE TEXT - TỔNG QUÁT CHO Y TẾ
# ==============================================================================

# Medical abbreviations và expansions phổ biến
MEDICAL_ABBREVIATIONS = {
    # Tiếng Việt
    "btm": "bệnh thận mạn",
    "tha": "tăng huyết áp",
    "đtđ": "đái tháo đường",
    "stn": "suy thận",
    "nmct": "nhồi máu cơ tim",
    "tbmn": "tai biến mạch não",
    "copd": "bệnh phổi tắc nghẽn mạn tính",
    "xn": "xét nghiệm",
    "bn": "bệnh nhân",
    "cđ": "chẩn đoán",
    "đt": "điều trị",
    "ls": "lâm sàng",
    "cls": "cận lâm sàng",
    "bs": "bác sĩ",
    
    # English abbreviations in Vietnamese medical docs
    "gfr": "độ lọc cầu thận",
    "egfr": "độ lọc cầu thận ước tính",
    "ckd": "bệnh thận mạn",
    "aki": "tổn thương thận cấp",
    "acei": "thuốc ức chế men chuyển",
    "arb": "thuốc chẹn thụ thể angiotensin",
    "nsaid": "thuốc kháng viêm không steroid",
    "ppi": "thuốc ức chế bơm proton",
    "hba1c": "hemoglobin a1c",
    "ldl": "cholesterol xấu",
    "hdl": "cholesterol tốt",
    "bp": "huyết áp",
    "hr": "nhịp tim",
    "bmi": "chỉ số khối cơ thể",
    "iv": "tiêm tĩnh mạch",
    "po": "đường uống",
    "bid": "hai lần mỗi ngày",
    "tid": "ba lần mỗi ngày",
    "qd": "mỗi ngày một lần",
    "prn": "khi cần",
}

# Synonyms - Gộp các từ đồng nghĩa về 1 dạng chuẩn
MEDICAL_SYNONYMS = {
    # Bệnh thận
    "bệnh thận mạn": ["bệnh thận mãn", "suy thận mạn", "suy thận mãn", "ckd", "bệnh thận mạn tính"],
    "tổn thương thận cấp": ["suy thận cấp", "aki", "tổn thương thận cấp tính"],
    
    # Tiểu đường
    "đái tháo đường": ["tiểu đường", "đtđ", "diabetes", "bệnh tiểu đường"],
    "đái tháo đường type 2": ["tiểu đường type 2", "đtđ típ 2", "đtđ type 2"],
    
    # Huyết áp
    "tăng huyết áp": ["cao huyết áp", "huyết áp cao", "tha", "hypertension"],
    
    # Tim mạch
    "nhồi máu cơ tim": ["nmct", "heart attack", "đau tim"],
    "suy tim": ["heart failure", "suy tim sung huyết"],
    
    # Thuốc
    "thuốc ức chế men chuyển": ["acei", "ace inhibitor", "thuốc ức chế ace"],
    "thuốc chẹn thụ thể angiotensin": ["arb", "angiotensin receptor blocker"],
    "thuốc lợi tiểu": ["lợi tiểu", "diuretic", "thuốc tiểu"],
    
    # Xét nghiệm
    "xét nghiệm máu": ["xn máu", "blood test"],
    "xét nghiệm nước tiểu": ["xn nước tiểu", "urinalysis"],
    "sinh thiết": ["biopsy", "sinh thiết thận"],
}

# Stopwords y tế (không normalize, giữ nguyên nghĩa)
MEDICAL_STOPWORDS = {
    "bệnh", "thuốc", "triệu chứng", "điều trị", "xét nghiệm",
    "chẩn đoán", "phương pháp", "kỹ thuật", "quy trình"
}


def normalize_medical_text(text: str, expand_abbrev: bool = True, 
                           use_synonyms: bool = True) -> str:
    """
    Normalize text cho domain y tế - TỔNG QUÁT
    
    Args:
        text: Input text
        expand_abbrev: Mở rộng viết tắt không
        use_synonyms: Gộp synonyms không
    
    Returns:
        Normalized text
    """
    if not text:
        return "Unknown"
    
    # Step 1: Unicode normalization
    text = unicodedata.normalize("NFC", text)
    
    # Step 2: Lowercase
    text = text.strip().lower()
    
    # Step 3: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Step 4: Expand abbreviations
    if expand_abbrev:
        words = text.split()
        expanded = []
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in MEDICAL_ABBREVIATIONS:
                expanded.append(MEDICAL_ABBREVIATIONS[clean_word])
            else:
                expanded.append(word)
        text = ' '.join(expanded)
    
    # Step 5: Normalize synonyms
    if use_synonyms:
        for canonical, variants in MEDICAL_SYNONYMS.items():
            for variant in variants:
                if variant in text:
                    text = text.replace(variant, canonical)
    
    # Step 6: Remove common noise patterns
    noise_patterns = [
        r'\([^)]*\)',  # Remove parenthetical notes
        r'\[[^\]]*\]',  # Remove bracketed notes
        r'\d+\.\d+',    # Remove version numbers
        r'trang \d+',   # Remove page references
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text)
    
    # Step 7: Final cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 8: Title case (giữ nguyên tiếng Việt)
    return text.title()


def extract_medical_entities_simple(text: str) -> List[str]:
    """
    Quick rule-based entity extraction (backup khi LLM fail)
    """
    entities = []
    
    # Pattern matching cho entities y tế phổ biến
    patterns = [
        r'bệnh\s+[\w\s]+',
        r'thuốc\s+[\w\s]+',
        r'triệu chứng\s+[\w\s]+',
        r'hội chứng\s+[\w\s]+',
        r'viêm\s+[\w\s]+',
        r'suy\s+[\w\s]+',
        r'ung thư\s+[\w\s]+',
        r'nhiễm\s+[\w\s]+',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        entities.extend([m.strip() for m in matches])
    
    return list(set(entities))


# ==============================================================================
# PHẦN 2: OPTIMIZED PROMPTS FOR ENTITY EXTRACTION
# ==============================================================================

# Prompt chính TÍCH HỢP extraction entities VÀ relations
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


# Prompt cho Bidirectional Relation với examples
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

## EXAMPLE:
Entity1: Đái tháo đường type 2
Entity2: Bệnh thận mạn
Context: "Đái tháo đường là nguyên nhân hàng đầu gây bệnh thận mạn ở Việt Nam."

Output:
- Forward: Đái tháo đường type 2 → CAUSES → Bệnh thận mạn (confidence=9)
- Backward: Bệnh thận mạn → COMPLICATION_OF → Đái tháo đường type 2 (confidence=8)
- Evidence: "nguyên nhân hàng đầu gây bệnh thận mạn"

## QUY TẮC:
1. PHẢI có cả quan hệ Forward VÀ Backward
2. Confidence score phải có căn cứ từ context
3. Evidence phải trích dẫn từ văn bản gốc"""),
    ("human", "Entity1: {entity1}\nEntity2: {entity2}\nContext: {context}")
])


# Prompt cho Entity Summarization
ENTITY_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tạo TÓM TẮT Y TẾ cho các thực thể.

## YÊU CẦU:
1. Mỗi entity có tóm tắt 2-3 câu DỰA TRÊN CONTEXT
2. Tập trung vào: định nghĩa, vai trò, tầm quan trọng trong y tế
3. Chấm điểm importance dựa trên mức độ quan trọng trong context

## EXAMPLE:
Entity: Bệnh thận mạn
Context: "Bệnh thận mạn là tình trạng suy giảm chức năng thận kéo dài > 3 tháng. BTM giai đoạn 5 cần điều trị thay thế thận."

Summary: Bệnh thận mạn là tình trạng thận mất dần chức năng lọc máu theo thời gian. Bệnh được chia thành 5 giai đoạn, giai đoạn cuối cần lọc máu hoặc ghép thận để duy trì sự sống.
Importance: 9

## LƯU Ý:
- KHÔNG thêm thông tin ngoài context
- KHÔNG dài quá 3 câu
- Ưu tiên thông tin LÂM SÀNG hơn là học thuật"""),
    ("human", "Entities: {entities}\nContext: {context}")
])


# ==============================================================================
# PHẦN 3: HELPER FUNCTIONS
# ==============================================================================

def get_entity_extraction_prompt():
    """Return optimized entity extraction prompt"""
    return ENTITY_EXTRACTION_PROMPT

def get_relation_extraction_prompt():
    """Return optimized relation extraction prompt"""
    return RELATION_EXTRACTION_PROMPT

def get_entity_summary_prompt():
    """Return optimized entity summary prompt"""
    return ENTITY_SUMMARY_PROMPT

def validate_entity(entity_name: str, entity_type: str) -> bool:
    """Validate if entity is valid medical entity"""
    # Skip empty or too short
    if not entity_name or len(entity_name) < 2:
        return False
    
    # Skip administrative content
    admin_patterns = [
        r'quyết định', r'văn bản', r'bộ y tế', r'trang \d+',
        r'điều \d+', r'khoản \d+', r'mục \d+', r'phụ lục'
    ]
    for pattern in admin_patterns:
        if re.search(pattern, entity_name.lower()):
            return False
    
    # Valid entity types
    valid_types = {
        'DISEASE', 'DRUG', 'SYMPTOM', 'TEST', 'ANATOMY',
        'TREATMENT', 'PROCEDURE', 'RISK_FACTOR', 'LAB_VALUE'
    }
    if entity_type.upper() not in valid_types:
        return False
    
    return True


def deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    """Remove duplicate entities based on normalized name"""
    seen = {}
    for ent in entities:
        norm_name = normalize_medical_text(ent.get('name', ''))
        if norm_name not in seen:
            seen[norm_name] = ent
        else:
            # Keep the one with higher relevance score
            if ent.get('relevance_score', 0) > seen[norm_name].get('relevance_score', 0):
                seen[norm_name] = ent
    return list(seen.values())


# Export all
__all__ = [
    'normalize_medical_text',
    'extract_medical_entities_simple',
    'get_entity_extraction_prompt',
    'get_relation_extraction_prompt', 
    'get_entity_summary_prompt',
    'validate_entity',
    'deduplicate_entities',
    'MEDICAL_ABBREVIATIONS',
    'MEDICAL_SYNONYMS',
]
