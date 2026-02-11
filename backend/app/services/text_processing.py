import re
import unicodedata
from typing import List, Dict

# ==============================================================================
# DATA CONSTANTS
# ==============================================================================

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

MEDICAL_SYNONYMS = {
    "bệnh thận mạn": ["bệnh thận mãn", "suy thận mạn", "suy thận mãn", "ckd", "bệnh thận mạn tính"],
    "tổn thương thận cấp": ["suy thận cấp", "aki", "tổn thương thận cấp tính"],
    "đái tháo đường": ["tiểu đường", "đtđ", "diabetes", "bệnh tiểu đường"],
    "đái tháo đường type 2": ["tiểu đường type 2", "đtđ típ 2", "đtđ type 2"],
    "tăng huyết áp": ["cao huyết áp", "huyết áp cao", "tha", "hypertension"],
    "nhồi máu cơ tim": ["nmct", "heart attack", "đau tim"],
    "suy tim": ["heart failure", "suy tim sung huyết"],
    "thuốc ức chế men chuyển": ["acei", "ace inhibitor", "thuốc ức chế ace"],
    "thuốc chẹn thụ thể angiotensin": ["arb", "angiotensin receptor blocker"],
    "thuốc lợi tiểu": ["lợi tiểu", "diuretic", "thuốc tiểu"],
    "xét nghiệm máu": ["xn máu", "blood test"],
    "xét nghiệm nước tiểu": ["xn nước tiểu", "urinalysis"],
    "sinh thiết": ["biopsy", "sinh thiết thận"],
}

MEDICAL_STOPWORDS = {
    "bệnh", "thuốc", "triệu chứng", "điều trị", "xét nghiệm",
    "chẩn đoán", "phương pháp", "kỹ thuật", "quy trình"
}

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def normalize_medical_text(text: str, expand_abbrev: bool = True, 
                           use_synonyms: bool = True) -> str:
    """Normalize text cho domain y tế."""
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

def validate_entity(entity_name: str, entity_type: str) -> bool:
    """Validate if entity is valid medical entity."""
    if not entity_name or len(entity_name) < 2:
        return False
    
    admin_patterns = [
        r'quyết định', r'văn bản', r'bộ y tế', r'trang \d+',
        r'điều \d+', r'khoản \d+', r'mục \d+', r'phụ lục'
    ]
    for pattern in admin_patterns:
        if re.search(pattern, entity_name.lower()):
            return False
    
    valid_types = {
        'DISEASE', 'DRUG', 'SYMPTOM', 'TEST', 'ANATOMY',
        'TREATMENT', 'PROCEDURE', 'RISK_FACTOR', 'LAB_VALUE'
    }
    if entity_type.upper() not in valid_types:
        return False
    
    return True

def deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    """Remove duplicate entities based on normalized name."""
    seen = {}
    for ent in entities:
        norm_name = normalize_medical_text(ent.get('name', ''))
        if norm_name not in seen:
            seen[norm_name] = ent
        else:
            if ent.get('relevance_score', 0) > seen[norm_name].get('relevance_score', 0):
                seen[norm_name] = ent
    return list(seen.values())

def extract_medical_entities_simple(text: str) -> List[str]:
    """Quick rule-based entity extraction (backup)."""
    entities = []
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
