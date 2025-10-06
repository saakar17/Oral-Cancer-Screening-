# Oral-Cancer-Screening-
# 🦷 AI-Assisted Oral Cancer Screening Web App
<img width="1470" height="920" alt="Screenshot 2025-10-06 at 22 26 15" src="https://github.com/user-attachments/assets/bd649479-419b-42c0-ad9b-c24472fa5512" />


Empowering early detection through AI-driven oral health screening

This web application provides an intelligent, user-friendly interface for screening oral cavity images using deep learning model (YOLO-based). It assists clinicians and community health workers in detecting oral abnormalities such as **Benign Lesions**, **Oral Potentially Malignant Disorders (OPMD)**, and **Oral Cavity Carcinoma (OCA)** — directly from uploaded or live-captured images.

---

## 🚀 Key Features

- 🧠 **AI-Powered Detection** – Real-time analysis using YOLO deep learning models
  
- 📸 **Multiple Input Modes**  
  - *Single Image Mode* – Upload and analyze one image.  
  - *Batch Analysis Mode* – Upload multiple images or a ZIP folder for bulk screening.  
  - *Live Mode* – Capture directly via webcam or iPhone Continuity Camera.
    
- 📊 **Automatic Report Generation** – Generates color-coded risk assessments and downloadable medical PDF reports.
  
- 🌐 **Bilingual Support** – English report text with Nepali impressions and visual guidance.
  
- 🧾 **Batch Export** – Download all findings as CSV or PDF in one click.
  
- 🖼️ **Interactive Visualization** – Displays annotated images highlighting detected lesions.


---

## ⚙️ Installation Guide

### 1️⃣ Clone the Repository


### 2️⃣ Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Add the YOLO Model
Place your trained YOLO model file (`best.pt`) inside:
```
app/web_app/best.pt
```

### 5️⃣ Run the Web App
```bash
streamlit run app.py
```

Then open your browser at:
```
http://localhost:8501
```

---

## 🧠 AI Model Details

- **Model Type:** YOLOv8 (custom trained for oral lesion detection)
- **Classes:** `Healthy`, `Benign`, `OPMD`, `OCA`
- **Frameworks Used:** PyTorch, OpenCV, Streamlit, ReportLab
- **Device Support:** GPU (CUDA) & CPU fallback

---

## 📋 Example Output

| AI Diagnosis | Confidence | Risk Assessment | Clinical Recommendation |
|---------------|-------------|------------------|--------------------------|
| Oral Cavity Carcinoma | 96.7% | 🔴 High Risk | Immediate oncology consultation |
| OPMD | 84.3% | 🟠 Medium Risk | Specialist referral and biopsy consideration |
| Benign Lesion | 92.1% | 🟡 Low Risk | Routine follow-up recommended |
| Healthy | 99.2% | 🟢 No Risk | Maintain oral hygiene |

---

## 🏗️ Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend | [Streamlit](https://streamlit.io) |
| AI Engine | YOLOv8 (PyTorch) |
| Image Processing | OpenCV, NumPy, PIL |
| Report Generation | ReportLab |
| Data Management | Pandas |
| Deployment | Localhost / Streamlit Cloud / Custom Server |

---

## 🧾 Generated Reports

Each analysis automatically generates:
- Annotated lesion image
- Clinical summary (Diagnosis, Risk, Referral)
- English + Nepali impression
- Downloadable PDF report

---

###  _“Early detection saves lives — with AI, we can make it accessible to everyone.”_
