# 📚 Hướng dẫn Chi tiết 20 Loại Quan hệ Y tế

## 🔄 So sánh: 10 → 20 Loại Quan hệ

### Trước (10 loại):
```
1. CAUSES
2. HAS_SYMPTOM
3. INDICATES
4. TREATS
5. REQUIRES_TEST
6. CONTRAINDICATED
7. INCREASES_RISK
8. COMPLICATION_OF
9. ASSOCIATED_WITH
10. WORSENS
```

### Sau (20 loại - +10 loại MỚI):
```
✅ Cũ:
1. CAUSES
2. HAS_SYMPTOM
3. INDICATES
4. TREATS
5. REQUIRES_TEST
6. CONTRAINDICATED
7. INCREASES_RISK
8. COMPLICATION_OF
9. ASSOCIATED_WITH
10. WORSENS

🆕 Mới:
11. MANAGE_WITH ⭐
12. DOSAGE_ADJUSTMENT ⭐
13. CONTRAINDICATED_WITH ⭐
14. MONITORING_REQUIRED ⭐
15. PREVENTS ⭐
16. METABOLIZED_BY ⭐
17. SIDE_EFFECT_OF ⭐
18. INTERACTION_WITH ⭐
19. STAGES ⭐
20. RISK_STRATIFICATION ⭐
```

---

## 📖 Giải thích Chi tiết Từng Loại

### GROUP 1: DISEASE - SYMPTOM/OUTCOME (3 loại)

#### 1️⃣ HAS_SYMPTOM
- **Định nghĩa**: Bệnh có triệu chứng
- **Hướng**: Bệnh → Triệu chứng
- **Ví dụ**:
  - Bệnh thận mạn → Phù chân
  - Đái tháo đường → Khát nước
  - Huyết áp cao → Đau đầu
- **Luôn trích**: Khi văn bản nói bệnh gây ra triệu chứng
- **Confidence**: 8-10 (rõ ràng) hoặc 5-6 (gợi ý)

#### 2️⃣ COMPLICATION_OF
- **Định nghĩa**: Biến chứng của bệnh (hệ quả nặng hơn)
- **Hướng**: Biến chứng → Bệnh gốc
- **Ví dụ**:
  - Tăng kali máu ← Bệnh thận
  - Loạn nhịp ← Bệnh thận
  - Suy tim ← Huyết áp cao (mạn tính)
- **Khác với HAS_SYMPTOM**: Complication = triệu chứng/bệnh nặng hơn
- **Confidence**: 8-10

#### 3️⃣ ASSOCIATED_WITH
- **Định nghĩa**: Liên quan với nhau (không rõ nhân quả)
- **Hướng**: Hai chiều (A ↔ B)
- **Ví dụ**:
  - Tiểu đường ↔ Béo phì
  - Bệnh thận ↔ Thiếu máu
  - Huyết áp cao ↔ Bệnh tim
- **Khi dùng**: Khi chỉ nói "liên quan" không rõ gây-gây
- **Confidence**: 5-8

---

### GROUP 2: CAUSATION & RISK (4 loại)

#### 4️⃣ CAUSES
- **Định nghĩa**: Gây ra trực tiếp (nhân quả rõ ràng)
- **Hướng**: Nguyên nhân → Bệnh/triệu chứng
- **Ví dụ**:
  - Tiểu đường → Bệnh thận mạn
  - Huyết áp cao → Bệnh thận mạn
  - Hút thuốc → Ung thư phổi
- **Confidence**: 9-10 (rõ nhất)
- **Khác với INCREASES_RISK**: CAUSES = chắc chắn, INCREASES_RISK = tăng nguy cơ

#### 5️⃣ INCREASES_RISK
- **Định nghĩa**: Tăng nguy cơ bệnh (yếu tố nguy cơ)
- **Hướng**: Yếu tố → Bệnh
- **Ví dụ**:
  - Hút thuốc → ↑ nguy cơ ung thư
  - GFR < 60 → ↑ nguy cơ ung thư thận
  - Béo phì → ↑ nguy cơ tiểu đường
- **Confidence**: 7-9
- **Note**: Không chắc chắn 100%, nhưng tăng xác suất

#### 6️⃣ WORSENS
- **Định nghĩa**: Làm nặng tình trạng hiện có
- **Hướng**: Yếu tố → Bệnh (làm nặng)
- **Ví dụ**:
  - Tăng kali → Loạn nhịp tim
  - Các nhiễm khuẩn → Suy thận (nặng hơn)
  - Không tuân thủ → Bệnh thận (tiến triển nhanh)
- **Confidence**: 7-9

#### 7️⃣ PREVENTS
- **Định nghĩa**: Phòng ngừa bệnh
- **Hướng**: Biện pháp → Bệnh
- **Ví dụ**:
  - Vaccin cúm → Phòng Pneumonia
  - Thay thận → Phòng tử vong do suy thận
  - Thuốc hạ cholesterol → Phòng suy tim
- **Confidence**: 7-10 (tùy chứng cứ)

---

### GROUP 3: DIAGNOSIS & MONITORING (5 loại)

#### 8️⃣ INDICATES
- **Định nghĩa**: Chỉ định chẩn đoán bệnh
- **Hướng**: Triệu chứng/Test → Bệnh
- **Ví dụ**:
  - GFR < 15 → Bệnh thận giai đoạn 5
  - Protein niệu > 3g/24h → Hội chứng thận hư
  - Ho > 3 tuần → Lao
- **Confidence**: 8-10

#### 9️⃣ REQUIRES_TEST
- **Định nghĩa**: Bệnh cần kiểm tra giám sát
- **Hướng**: Bệnh → Test
- **Ví dụ**:
  - Bệnh thận → Xét nghiệm GFR
  - Tiểu đường → Xét nghiệm glucose
  - Huyết áp cao → Đo huyết áp hàng ngày
- **Confidence**: 8-9

#### 🔟 MONITORING_REQUIRED
- **Định nghĩa**: Cần theo dõi liên tục (kèm tần suất)
- **Hướng**: Tình trạng → Test (liên tục)
- **Ví dụ**:
  - Bệnh thận giai đoạn 4 → GFR (3 tháng/lần)
  - Trên EPO → Hemoglobin (hàng tháng)
  - Trên Warfarin → INR (hàng tháng)
- **Evidence**: Nên bao gồm tần suất
- **Confidence**: 8-9

#### 1️⃣1️⃣ STAGES (Mới)
- **Định nghĩa**: Giai đoạn của bệnh
- **Hướng**: Giai đoạn cụ thể → Bệnh chung
- **Ví dụ**:
  - Bệnh thận mạn giai đoạn 5 → Stage of: Bệnh thận mạn
  - CKD giai đoạn 3b → Stage of: CKD
  - CHF NYHA IV → Stage of: Suy tim
- **Confidence**: 9-10

#### 1️⃣2️⃣ RISK_STRATIFICATION (Mới)
- **Định nghĩa**: Phân tầng nguy cơ (kết hợp nhiều yếu tố)
- **Hướng**: Yếu tố → Mức nguy cơ
- **Ví dụ**:
  - GFR 15-29 + Protein niệu ↑ → HIGH RISK progression
  - eGFR < 15 + Reanin ↑ → HIGH RISK ESRD
  - Triple positive → HIGH RISK tử vong
- **Note**: Thường dùng cho prognosis
- **Confidence**: 7-9

---

### GROUP 4: TREATMENT (5 loại)

#### 1️⃣3️⃣ TREATS
- **Định nghĩa**: Thuốc/liệu pháp điều trị bệnh
- **Hướng**: Thuốc → Bệnh
- **Ví dụ**:
  - Erythropoietin → Thiếu máu
  - Lisinopril → Bệnh thận (chậm tiến triển)
  - Thay thận → Suy thận giai đoạn 5
- **Confidence**: 8-10

#### 1️⃣4️⃣ MANAGE_WITH (Mới ⭐)
- **Định nghĩa**: Quản lý/điều trị kết hợp với các biện pháp khác
- **Hướng**: Bệnh/Triệu chứng → Kết hợp thuốc/liệu pháp
- **Ví dụ**:
  - Thiếu máu do suy thận → MANAGE_WITH: EPO + Sắt + Folic acid
  - Huyết áp cao → MANAGE_WITH: ACE-I + Calcium blocker
  - Bệnh thận → MANAGE_WITH: Giúp đỡ toàn diện + Chế độ ăn
- **Khác với TREATS**: 
  - TREATS = một thuốc điều trị
  - MANAGE_WITH = chiến lược điều trị toàn diện
- **Confidence**: 7-9

#### 1️⃣5️⃣ SIDE_EFFECT_OF (Mới ⭐)
- **Định nghĩa**: Tác dụng phụ của thuốc
- **Hướng**: Tác dụng phụ ← Thuốc
- **Ví dụ**:
  - Ho khô ← Lisinopril (ACE-I)
  - Tăng kali ← ACE-I/ARB
  - Phù mặt ← Amlodipine
- **Note**: Tệp này QUAN TRỌNG cho độc tính thếp
- **Confidence**: 8-10

#### 1️⃣6️⃣ DOSAGE_ADJUSTMENT (Mới ⭐)
- **Định nghĩa**: Cần điều chỉnh liều theo tình trạng
- **Hướng**: Yếu tố → Thuốc (điều chỉnh liều)
- **Ví dụ**:
  - GFR < 30 → Gentamicin (giảm 50% liều)
  - CrCl < 50 → Metformin (tránh/giảm liều)
  - Tuổi > 70 → ACE-I (bắt đầu liều thấp)
- **Evidence**: Nên bao gồm % điều chỉnh hoặc khuyến cáo
- **Confidence**: 8-10 (khi rõ chỉ định)

#### 1️⃣7️⃣ CONTRAINDICATED
- **Định nghĩa**: Chống chỉ định = KHÔNG dùng danhthuốc
- **Hướng**: Tình trạng → Thuốc (tránh)
- **Ví dụ**:
  - Suy thận nặng → Metformin (CONTRAINDICATED)
  - Rối loạn kali → ACE-I (CONTRAINDICATED)
  - Hemoglobin > 12 → EPO (CONTRAINDICATED)
- **Confidence**: 9-10 (nghiêm trọng)

---

### GROUP 5: DRUG-DRUG INTERACTIONS (3 loại)

#### 1️⃣8️⃣ CONTRAINDICATED_WITH (Mới ⭐)
- **Định nghĩa**: Chống chỉ định kết hợp (thuốc-thuốc)
- **Hướng**: Thuốc A ↔ Thuốc B (tránh kết hợp)
- **Ví dụ**:
  - ACE-I ↔ ARB (CONTRAINDICATED_WITH)
  - Warfarin ↔ Aspirin (mạnh) (CONTRAINDICATED_WITH)
  - Simvastatin ↔ Erythromycin (CONTRAINDICATED_WITH)
- **Evidence**: Nên giải thích lý do
- **Confidence**: 8-10

#### 1️⃣9️⃣ INTERACTION_WITH (Mới ⭐)
- **Định nghĩa**: Tương tác (hệ quả cụ thể)
- **Hướng**: Thuốc A + Thuốc B → Hệ quả
- **Ví dụ**:
  - Simvastatin + Erythromycin → ↑ Rhabdomyolysis
  - Warfarin + NSAIDs → ↑ Chảy máu
  - Methotrexate + NSAIDs → ↑ Tổn thương thận
- **Khác với CONTRAINDICATED_WITH**: 
  - CONTRAINDICATED = tránh hoàn toàn
  - INTERACTION = hệ quả cụ thể, có thể dùng cẩn thận
- **Confidence**: 7-9

#### 2️⃣0️⃣ METABOLIZED_BY (Mới ⭐)
- **Định nghĩa**: Được chuyển hóa bởi enzyme
- **Hướng**: Thuốc → Enzyme (CYP450, κ.v.)
- **Ví dụ**:
  - Warfarin → Metabolized by: CYP2C9
  - Simvastatin → Metabolized by: CYP3A4
  - Clopidogrel → Metabolized by: CYP2C19
- **Ý nghĩa**: Để dự báo tương tác
- **Confidence**: 9-10

---

## 💡 Hội chứng Y tế Toàn diện (Ví dụ)

Cùng một ca bệnh, tất cả 20 quan hệ:

**Tình huống**: "Bệnh nhân nữ 65 tuổi, CKD giai đoạn 5 (GFR=12) với tiểu đường type 2, thiếu máu (Hb=8), được điều trị bằng EPO 4000 unit/week, Lisinopril, Metformin."

```
Entities: 
- Bệnh thận mạn giai đoạn 5 (DISEASE)
- Đái tháo đường type 2 (DISEASE)
- Thiếu máu (DISEASE)
- GFR=12 (LAB_VALUE)
- Hb=8 (LAB_VALUE)
- EPO 4000 unit/week (DRUG)
- Lisinopril (DRUG)
- Metformin (DRUG)

Relations:
 1️⃣ CAUSES: Đái tháo đường → Bệnh thận (gây ra trực tiếp)
 2️⃣ CAUSES: Huyết áp cao → Bệnh thận
 3️⃣ HAS_SYMPTOM: Bệnh thận → Phù chân
 4️⃣ COMPLICATION_OF: Thiếu máu ← Bệnh thận
 5️⃣ INDICATES: GFR=12 → Bệnh thận giai đoạn 5
 6️⃣ REQUIRES_TEST: Bệnh thận → Xét nghiệm GFR
 7️⃣ MONITORING_REQUIRED: Thiếu máu → Hemoglobin (hàng tháng)
 8️⃣ STAGES: CKD giai đoạn 5 → Stage of: Bệnh thận mạn
 9️⃣ RISK_STRATIFICATION: GFR=12 + Protein↑ → HIGH RISK ESRD
🔟 TREATS: EPO → Thiếu máu
1️⃣1️⃣ MANAGE_WITH: Thiếu máu → EPO + Sắt + Folic acid
1️⃣2️⃣ SIDE_EFFECT_OF: Ho khô ← Lisinopril
1️⃣3️⃣ DOSAGE_ADJUSTMENT: GFR < 30 → Metformin (giảm liều 50%)
1️⃣4️⃣ CONTRAINDICATED: GFR=12 → Metformin (tránh)
1️⃣5️⃣ CONTRAINDICATED_WITH: ACE-I ↔ ARB (nếu dùng cả 2)
1️⃣6️⃣ INTERACTION_WITH: Metformin + NSAID → ↑ Acid lactic
1️⃣7️⃣ METABOLIZED_BY: Lisinopril → Kidney (bài tiết)
1️⃣8️⃣ PREVENTS: Thay thận → Phòng tử vong
1️⃣9️⃣ ASSOCIATED_WITH: CKD ↔ Tăng huyết áp
2️⃣0️⃣ WORSENS: Không tuân thủ → CKD (tiến triển nhanh)
```

---

## ✅ Checklist: Các quan hệ nên trích

- [ ] **CAUSES** - Luôn trích nếu nói gây bệnh
- [ ] **HAS_SYMPTOM** - Luôn trích nếu nói triệu chứng
- [ ] **TREATS** - Luôn trích nếu nói điều trị
- [ ] **INDICATES** - Luôn trích nếu nói chẩn đoán
- [ ] **CONTRAINDICATED** - Luôn trích nếu nói chống chỉ định
- [ ] **SIDE_EFFECT_OF** - Luôn trích nếu nói tác dụng phụ
- [ ] **DOSAGE_ADJUSTMENT** - Luôn trích nếu nói điều chỉnh liều
- [ ] **MONITORING_REQUIRED** - Trích nếu nói theo dõi + tần suất
- [ ] **COMPLICATION_OF** - Trích nếu nói biến chứng
- [ ] **REQUIRES_TEST** - Trích nếu nói cần kiểm tra
- [ ] **INCREASES_RISK** - Trích nếu nói tăng nguy cơ
- [ ] **CONTRAINDICATED_WITH** - Trích nếu nói tránh dùng cùng
- [ ] **MANAGE_WITH** - Trích nếu nói điều trị kết hợp
- [ ] **INTERACTION_WITH** - Trích nếu nói tương tác hóa chất
- [ ] **METABOLIZED_BY** - Trích nếu nói chuyển hóa enzyme
- [ ] **PREVENTS** - Trích nếu nói phòng ngừa
- [ ] **WORSENS** - Trích nếu nói làm nặng
- [ ] **ASSOCIATED_WITH** - Trích nếu nói liên quan mơ hồ
- [ ] **STAGES** - Trích nếu nói giai đoạn bệnh
- [ ] **RISK_STRATIFICATION** - Trích nếu nói phân tầng nguy cơ

---

## 🎯 Kết luận

Bạn đã nâng cấp từ **10 → 20 loại quan hệ**. Điều này cho phép:

✅ Trích xuất **chi tiết hơn** về tương tác thuốc
✅ Hỗ trợ **kiến thức y tế toàn diện** (chẩn đoán → điều trị → giám sát)
✅ Xây dựng **đồ thị y tế chuyên sâu** cho y bác sĩ
✅ Phát hiện **contraindication & interactions** tự động

**Đề xuất tiếp theo**: Có thể thêm các quan hệ nếu cần:
- PALLIATIVE_CARE (chăm sóc giảm nhẹ)
- LIFESTYLE_MODIFICATION (thay đổi lối sống)
- SCREENING_FOR (sàng lọc)
- ALTERNATIVE_TO (thay thế)
