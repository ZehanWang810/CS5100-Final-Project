# CS5100 Final Project — Student Success Prediction

This repository contains my implementation for the **CS5100 Final Project (Phase 1 & Phase 2)**.  
The goal is to predict whether a student is *at risk* of failing using the UCI Student Performance dataset.

---

## 📂 Repository Structure
Final/
│
├── student_project/          # Phase 1 implementation (preprocessing + ML pipeline)
│
├── tests/                   # Provided autograder tests
│
├── datasets/                # Student dataset
│
├── phase2_scope1_full_data.py        # Scope Item 1 — Full dataset evaluation
├── phase2_scope2_feature_selection.py # Scope Item 2 — Feature selection (Mutual Info)
├── phase2_scope3_stacking.py          # Scope Item 3 — Stacking ensemble
│
└── CS5100Final_Report.pdf             # Full report (Phase 1 + Phase 2 + Reflection)
---

## 🌟 Phase 1 Summary

Phase 1 implements the full baseline pipeline:

- Data loading + validation  
- Preprocessing (one-hot encoding, scaling, leakage removal)
- Target construction (`at_risk`)
- Summary statistics & correlation matrix  
- Gradient Boosting baseline  
- **Custom Random Forest implementation from scratch**  
- All **Phase 1 tests pass**:

---

## 🌟 Phase 2 — Scope Items (Total = 4 points for solo student)

### ✔ **Scope Item 1 — Use full dataset (1 point)**
Re-trained models using all 395 samples.  
Significant improvement in both F1 and ROC-AUC.

### ✔ **Scope Item 2 — Feature selection (1 point)**
Selected Top-K features using Mutual Information.  
Improved Gradient Boosting performance (F1 ↑).

### ✔ **Scope Item 3 — Stacking ensemble (2 points)**
Stacked Gradient Boosting + Random Forest using Logistic Regression as a meta-learner.  
Demonstrated correct ensemble implementation.

---

## ▶️ Running the Code

### **Phase 1 tests**

pytest tests/test_phase_1.py -q

### **Phase 2 scripts**
python phase2_scope1_full_data.py
python phase2_scope2_feature_selection.py
python phase2_scope3_stacking.py

---

## 🧪 Environment
- Python 3.11+
- numpy
- pandas
- scikit-learn

---

## 📄 Report
The full report (Phase 1 + Phase 2 + Reflection + Appendix) is included here:

📎 **CS5100Final_Report.pdf**

---

## 🔗 Author
**Zehan Wang**  
Northeastern University  
CS5100 Foundations of AI — Final Project
