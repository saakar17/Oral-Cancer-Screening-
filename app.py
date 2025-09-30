import streamlit as st
import os
import sys
import tempfile
import zipfile
import time
from typing import List, Dict
import pandas as pd
import numpy as np
import cv2
import torch
from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from yolo_inference import YOLOInferenceEngine
def show_live_mode(engine: YOLOInferenceEngine):
    st.markdown("### Live Mode")
    col_p1, col_p2, col_p3 = st.columns([2,1,1])
    with col_p1:
        patient_name = st.text_input("Patient Name", placeholder="Enter patient name", key="live_patient_name")
    with col_p2:
        patient_age = st.number_input("Age", min_value=0, max_value=120, step=1, value=0, key="live_patient_age")
    with col_p3:
        patient_id = st.text_input("Patient ID", placeholder="Optional", key="live_patient_id")

    # Open iPhone Continuity Camera and capture a frame
    cap = cv2.VideoCapture(0)  # iPhone Continuity Camera
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        st.error("Failed to access Continuity Camera. Make sure your iPhone is unlocked and nearby.")
        return

    # Display the captured image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, channels="RGB", width=400)

    # Save captured frame
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        captured_image_path = tmp.name

    # Proceed with AI analysis and PDF/report just like single image mode
    run_detection = st.button('Run AI Analysis', type='primary', use_container_width=True)
    if run_detection:
        if not patient_name:
            st.error("Please enter patient name before running analysis.")
            return
        try:
            with st.spinner("🔬 Analyzing image..."):
                result = engine.predict_single_image(
                    captured_image_path,
                    conf_threshold=DEFAULT_CONF,
                    iou_threshold=DEFAULT_IOU
                )

                detections = result.get('detections', []) or []
                if len(detections) > 1:
                    best_det = max(detections, key=lambda d: d.get('confidence', 0))
                    result['detections'] = [best_det]

                original_image = result.get('original_image', None)
                if original_image is None:
                    st.error("Invalid image format")
                    return

                annotated_image = draw_detections_rgb(original_image, result.get('detections', [])) if result.get('detections') else original_image
                st.markdown("### Analysis Results")
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    st.image(annotated_image, caption=f'AI Analysis - Patient: {patient_name}', width=400)

            # Generate PDF report
            with st.spinner("Preparing report for download..."):
                rows = results_to_report_rows([result], include_patient_fields=True,
                                              patient_name=patient_name, patient_age=patient_age, patient_id=patient_id)
                df = pd.DataFrame(rows)
                st.dataframe(df.style.apply(highlight_risk, axis=1), use_container_width=True, hide_index=True)

                pdf_file_path = f"medical_report_{patient_name.replace(' ','_')}_{int(time.time())}.pdf"
                generate_pdf_single(
                    patient_info={'name':patient_name,'age':patient_age,'id':patient_id},
                    report_df=df,
                    annotated_image=annotated_image,
                    file_path=pdf_file_path
                )

                col_d1, col_d2, col_d3 = st.columns([1,1,1])
                with col_d2:
                    with open(pdf_file_path,'rb') as f:
                        st.download_button(
                            label="Download Medical Report (PDF)",
                            data=f.read(),
                            file_name=pdf_file_path,
                            mime='application/pdf',
                            use_container_width=True
                        )
        finally:
            try: os.unlink(captured_image_path)
            except Exception: pass
 
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.platypus import Spacer
 
APP_DIR = "/Users/saakar/AI-Assisted-Disease-Screening-for-Oral-Cancer/app/apps"
MODEL_PATH = os.path.join(APP_DIR, 'best.pt')
LOGO_PATH = os.path.join(APP_DIR, 'logo.png')
CLASS_NAMES = ['Healthy', 'Benign', 'OPMD', 'OCA']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
 
st.set_page_config(
   page_title='AI Assisted Disease Screening for Oral Cancer',
   page_icon='🦷',
   layout='wide',
   initial_sidebar_state='expanded'
)
 
# Professional header with logo - responsive design
col1, col2 = st.columns([0.1, 0.9])
with col1:
   st.image(LOGO_PATH, width=150)
with col2:
   st.markdown("""
   <div style='padding-top: 10px;'>
       <h1 style='margin: 0; color: #00bfff; font-weight: bold; font-size: 3rem; text-align: center;'>AI Assisted Oral Cancer Screening</h1>
       <p style='margin: 5px 0 0 0; color: #cccccc; font-size: 19px; text-align: center;'>एआईको सहयोगले मुखको क्यान्सर पत्ता लगाउने सुविधा</p>
   </div>
   """, unsafe_allow_html=True)
 
st.markdown("---")
 
# ------------------- Load Model -------------------
@st.cache_resource(show_spinner=False)
def load_engine():
   if not os.path.exists(MODEL_PATH):
       raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Place best.pt in the app folder.")
   engine = YOLOInferenceEngine(model_path=MODEL_PATH, class_names=CLASS_NAMES, device=DEVICE)
   return engine
 
# ------------------- Draw Detections -------------------
def draw_detections_rgb(image_rgb: np.ndarray, detections: List[Dict]) -> np.ndarray:
   annotated = image_rgb.copy()
   colors_list = [
       (34, 139, 34),   # Forest Green for Healthy
       (255, 165, 0),   # Orange for Benign
       (255, 69, 0),    # Red-Orange for OPMD
       (220, 20, 60),   # Crimson for OCA
   ]
   for det in detections:
       x1, y1, x2, y2 = det['bbox']
       conf = det['confidence']
       cid = det['class_id']
       cname = det['class_name']
       color = colors_list[cid % len(colors_list)]
       thickness = max(2, min(image_rgb.shape[:2]) // 300)
       cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
       label = f"{cname}: {conf*100:.1f}%"
       font_scale = max(0.6, min(image_rgb.shape[:2]) / 1200)
       font_thickness = max(1, thickness // 2)
       (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
       overlay = annotated.copy()
       cv2.rectangle(overlay, (x1, max(y1 - text_height - 8, 0)), (x1 + text_width + 8, y1), color, -1)
       annotated = cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0)
       cv2.putText(annotated, label, (x1 + 4, max(y1 - 4, text_height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
   return annotated
 
# ------------------- Referral Mapping & color map -------------------
referral_mapping = {
   'Healthy':'No Referral',
   'Benign':'Not Immediate',
   'OPMD':'Monitor / Specialist Referral',
   'OCA':'Urgent Referral'
}
 
risk_color_mapping = {
   'No Risk':'#90ee90',            # green
   'Low Risk':'#ffff99',           # yellow
   'Medium Risk - Requires Monitoring':'#ffd966',  # orange
   'High Risk - Immediate Consultation Required':'#ff9999' # red
}
 
# ------------------- Helpers -------------------
def get_patient_id_from_filename(filename: str) -> str:
   """Extract patient id from filename like K-025-03.jpg -> K-025"""
   base = os.path.splitext(os.path.basename(filename))[0]
   parts = base.split('-')
   if len(parts) >= 2:
       pid = '-'.join(parts[:-1])
   else:
       pid = base
   return pid
 
# ------------------- PDF Generation (Landscape, plain table) -------------------
def generate_pdf_single(patient_info: dict, report_df: pd.DataFrame, annotated_image: np.ndarray, file_path: str):
   """Generate a detailed single-patient PDF with header, patient info, insights, and annotated image."""
   from datetime import datetime

   doc = SimpleDocTemplate(file_path, pagesize=A4)
   elements = []
   styles = getSampleStyleSheet()

   # Custom styles using standard ReportLab fonts
   title_style = ParagraphStyle(
       name="TitleCenter",
       parent=styles['Title'],
       alignment=TA_CENTER,
       fontSize=18,
       spaceAfter=12,
       fontName="Helvetica-Bold"
   )
   right_style = ParagraphStyle(
       name="Right",
       parent=styles['Normal'],
       alignment=TA_RIGHT,
       fontSize=10,
       spaceAfter=8,
       fontName="Helvetica"
   )
   block_label_style = ParagraphStyle(
       name="BlockLabel",
       parent=styles['Normal'],
       fontSize=11,
       spaceAfter=4,
       leftIndent=4,
       fontName="Helvetica"
   )
   section_label_style = ParagraphStyle(
       name="SectionLabel",
       parent=styles['Heading3'],
       fontSize=13,
       spaceAfter=6,
       fontName="Helvetica-Bold"
   )
 
   # Title
   elements.append(Paragraph("AI Assisted Oral Cancer Screening Report", title_style))
 
   # Date/Time
   now = datetime.now()
   dt_str = now.strftime("Date: %Y-%m-%d<br/>Time: %H:%M:%S")
   elements.append(Paragraph(dt_str, right_style))
 
   # Patient block
   patient_name = patient_info.get('name', '')
   patient_id = patient_info.get('id', '')
   patient_age = patient_info.get('age', '')
   patient_info_block = (
       f"<b>Patient Name:</b> {patient_name}<br/>"
       f"<b>Patient ID:</b> {patient_id}<br/>"
       f"<b>Age:</b> {patient_age}"
   )
   elements.append(Paragraph(patient_info_block, block_label_style))
   elements.append(Spacer(1, 12))
 
   # Result label and image
   elements.append(Paragraph("Result:", section_label_style))
   tmp_img_path = None
   if annotated_image is not None:
       tmp_img_path = file_path.replace('.pdf','_temp.png')
       cv2.imwrite(tmp_img_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
       img = RLImage(tmp_img_path, width=260, height=230, hAlign='CENTER')
       elements.append(img)
       elements.append(Spacer(1, 10))
 
   # Insights from first row
   elements.append(Paragraph("Insights:", section_label_style))
   row = report_df.iloc[0] if not report_df.empty else {}
   diagnosis = row.get("AI Diagnosis", "") if isinstance(row, dict) else row.get("AI Diagnosis", "")
   confidence = row.get("Confidence Level", "") if isinstance(row, dict) else row.get("Confidence Level", "")
   risk = row.get("Risk Assessment", "") if isinstance(row, dict) else row.get("Risk Assessment", "")
   recommendation = row.get("Clinical Recommendation", "") if isinstance(row, dict) else row.get("Clinical Recommendation", "")
   referral = row.get("Referral", "") if isinstance(row, dict) else row.get("Referral", "")
   insights_lines = [
       f"<b>Diagnosis:</b> {diagnosis}",
       f"<b>Confidence:</b> {confidence}",
       f"<b>Risk Assessment:</b> {risk}",
       f"<b>Clinical Recommendation:</b> {recommendation}",
       f"<b>Referral to Hospital:</b> {referral}",
   ]
   for line in insights_lines:
       elements.append(Paragraph(line, block_label_style))
   elements.append(Spacer(1, 12))
 
   # Impression section (user-provided English + Nepali image)
   elements.append(Paragraph("Impression:", section_label_style))

   # Impression dictionary for English text
   english_impression_texts = {
       "Normal Tissue":
           "Your oral cavity looks normal, with no unusual spots or growths seen. "
           "This means the tissue appears healthy. No special action is needed right now, just continue good oral hygiene and routine check-ups.",
       "Benign Lesion":
           "We noticed a small lesion (a visible change in the mouth tissue), but its appearance suggests it is benign, meaning non-cancerous. "
           "This does not pose any immediate risk. We recommend simple monitoring through regular check-ups to ensure it remains unchanged.",
       "Oral Potentially Malignant Disorder":
           "A lesion (change in the mouth tissue) was found that matches the description of an Oral Potentially Malignant Disorder (OPMD). "
           "This is not cancer, but such lesions may sometimes turn cancerous over time. To be safe, we recommend seeing a specialist, who may suggest a biopsy (a small tissue test) to understand it better.",
       "Oral Cavity Carcinoma":
           "A lesion has been detected that looks highly suggestive of Oral Cavity Carcinoma, which is a form of mouth cancer. "
           "This needs quick attention. We strongly recommend visiting a specialist soon for confirmatory tests and to discuss treatment options as early as possible.",
   }
   # Nepali impression image mapping
   nepali_impression_images = {
       "Normal Tissue": "/Users/saakar/AI-Assisted-Disease-Screening-for-Oral-Cancer/app/apps/nepali_healthy.png",
       "Benign Lesion": "/Users/saakar/AI-Assisted-Disease-Screening-for-Oral-Cancer/app/apps/nepali_benign.png",
       "Oral Potentially Malignant Disorder": "/Users/saakar/AI-Assisted-Disease-Screening-for-Oral-Cancer/app/apps/nepali_opmd.png",
       "Oral Cavity Carcinoma": "/Users/saakar/AI-Assisted-Disease-Screening-for-Oral-Cancer/app/apps/nepali_oca.png"
   }

   # Get correct English impression text and Nepali image
   english_text = english_impression_texts.get(diagnosis, "N/A")
   nepali_img_path = nepali_impression_images.get(diagnosis)

   # Add English impression as Paragraph
   elements.append(Paragraph(english_text, block_label_style))
   elements.append(Spacer(1, 12))

   # Add Nepali image (if exists) below the English impression
   if nepali_img_path and os.path.exists(nepali_img_path):
       # width chosen for visual alignment, aspect ratio preserved
       img = RLImage(nepali_img_path, width=390, height=75,  hAlign='CENTER')
       elements.append(img)
       elements.append(Spacer(1, 10))
   else:
       # fallback: show a placeholder if image missing
       elements.append(Paragraph("(Nepali impression unavailable)", block_label_style))
       elements.append(Spacer(1, 10))
 
   doc.build(elements)
   if tmp_img_path and os.path.exists(tmp_img_path):
       os.remove(tmp_img_path)
 
def generate_pdf_batch(report_df: pd.DataFrame, file_path: str):
   doc = SimpleDocTemplate(file_path, pagesize=landscape(A4))
   elements = []
   styles = getSampleStyleSheet()
   elements.append(Paragraph("Batch Analysis Report", styles['Title']))
   elements.append(Spacer(1,12))
 
   # drop Status and Inference Time if present (we keep unified schema)
   pdf_df = report_df.drop(columns=[c for c in ['Status','Inference Time (s)'] if c in report_df.columns])
 
   if 'Referral' not in pdf_df.columns and 'AI Diagnosis' in pdf_df.columns:
       pdf_df['Referral'] = pdf_df['AI Diagnosis'].map(referral_mapping)
 
   col_count = len(pdf_df.columns)
   page_width = landscape(A4)[0] - 2*30
   default_width = page_width / max(1, col_count)
   col_widths = [default_width]*col_count
 
   data = [[Paragraph(str(c), styles['Normal']) for c in pdf_df.columns]]
   for row in pdf_df.values.tolist():
       data.append([Paragraph(str(cell), styles['Normal']) for cell in row])
 
   t = Table(data, colWidths=col_widths, repeatRows=1)
   ts = TableStyle([
       ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
       ('GRID', (0,0), (-1,-1), 0.5, colors.black),
       ('ALIGN',(0,0),(-1,-1),'CENTER'),
       ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
       ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold')
   ])
   t.setStyle(ts)
   elements.append(t)
   doc.build(elements)
 
# ------------------- Results -> standardized report rows -------------------
def results_to_report_rows(results: List[Dict], include_patient_fields: bool = False, patient_name: str = '', patient_age: int = 0, patient_id: str = '') -> List[Dict]:
    """
    Convert results (from single or batch) into standardized report rows.
    include_patient_fields: if True, include Patient Name/Age/Patient ID (used for single mode).
    For batch mode, patient_id must be extracted from filename and passed in patient_id param.
    """
    rows = []
    finding_counter = 1
    # mappings
    class_mapping = {'Healthy':'Normal Tissue','Benign':'Benign Lesion','OPMD':'Oral Potentially Malignant Disorder','OCA':'Oral Cavity Carcinoma'}
    risk_mapping = {'Healthy':'No Risk','Benign':'Low Risk','OPMD':'Medium Risk - Requires Monitoring','OCA':'High Risk - Immediate Consultation Required'}
    recommendation_mapping = {'Healthy':'Regular oral hygiene maintenance','Benign':'Routine follow-up recommended','OPMD':'Specialist referral and biopsy consideration','OCA':'Immediate oncology consultation required'}

    for r in results:
        # Filter detections: keep only highest confidence if more than one
        detections = r.get('detections', []) or []
        if len(detections) > 1:
            best_det = max(detections, key=lambda d: d.get('confidence', 0))
            r['detections'] = [best_det]
        img_path = r.get('image_path', '')
        base_name = os.path.basename(img_path) if img_path else ''
        extracted_pid = get_patient_id_from_filename(base_name) if base_name else ''
        # if result contains an explicit error
        if 'error' in r and r['error']:
            # create a single row marking analysis error (do not color)
            row = {
                'S.No': finding_counter,
                'Patient Name': patient_name if include_patient_fields else '',
                'Age': patient_age if include_patient_fields else '',
                'Patient ID': patient_id if include_patient_fields else (extracted_pid or ''),
                'AI Diagnosis': 'Analysis Error',
                'Confidence Level': '',
                'Risk Assessment': '',
                'Referral': '',
                'Clinical Recommendation': r.get('error', ''),
                'Date': time.strftime('%Y-%m-%d'),
                'Time': time.strftime('%H:%M:%S')
            }
            rows.append(row)
            finding_counter += 1
            continue

        detections = r.get('detections', []) or []
        if len(detections) == 0:
            # no detections -> report row with new values as per requirements
            row = {
                'S.No': finding_counter,
                'Patient Name': patient_name if include_patient_fields else '',
                'Age': patient_age if include_patient_fields else '',
                'Patient ID': patient_id if include_patient_fields else (extracted_pid or ''),
                'AI Diagnosis': 'No Key Detection',
                'Confidence Level': 'N/A',
                'Risk Assessment': 'N/A',
                'Referral': 'N/A',
                'Clinical Recommendation': 'N/A',
                'Date': time.strftime('%Y-%m-%d'),
                'Time': time.strftime('%H:%M:%S')
            }
            # Remove 'Patient ID' for single mode
            if include_patient_fields and 'Patient ID' in row:
                del row['Patient ID']
            rows.append(row)
            finding_counter += 1
            continue

        # one row per detection
        for det in detections:
            det_name = det.get('class_name', '')
            conf = det.get('confidence', 0.0)
            # Set Clinical Recommendation for OPMD as per requirement
            if det_name == 'OPMD':
                clinical_recommendation = 'Refer to specialist'
            else:
                clinical_recommendation = recommendation_mapping.get(det_name, 'Consult specialist')
            # Build row with 'Referral' before 'Clinical Recommendation'
            row = {
                'S.No': finding_counter,
                'Patient Name': patient_name if include_patient_fields else '',
                'Age': patient_age if include_patient_fields else '',
                'Patient ID': patient_id if include_patient_fields else (extracted_pid or ''),
                'Class Name': det_name,
                'AI Diagnosis': class_mapping.get(det_name, det_name),
                'Confidence Level': f"{conf*100:.1f}%",
                'Risk Assessment': risk_mapping.get(det_name, 'Unknown'),
                'Referral': referral_mapping.get(det_name, 'Consult specialist'),
                'Clinical Recommendation': clinical_recommendation,
                'Date': time.strftime('%Y-%m-%d'),
                'Time': time.strftime('%H:%M:%S')
            }
            # Remove 'Patient ID' for single mode
            if include_patient_fields and 'Patient ID' in row:
                del row['Patient ID']
            rows.append(row)
            finding_counter += 1

    return rows


 
# ------------------- Highlight function for Streamlit -------------------
def highlight_risk(row):
   """
   Returns a list of CSS strings for the given row.
   Skip highlighting when AI Diagnosis indicates error or Confidence is empty/N/A.
   """
   ai_diag = str(row.get('AI Diagnosis', '')).lower()
   conf = str(row.get('Confidence Level', '')).strip()
   # skip coloring for analysis errors or empty confidence
   if 'error' in ai_diag or conf == '' or conf.upper() == 'N/A':
       return [''] * len(row)
   risk = str(row.get('Risk Assessment', 'No Risk'))
   color = risk_color_mapping.get(risk, "#a1f09a")
   # return same style for all columns in row
   return [f'background-color: {color}; color: black'] * len(row)
 
# ------------------- Single Image Mode -------------------
def show_single_image(engine: YOLOInferenceEngine):
   st.markdown("### Single Image Analysis")
   col_p1, col_p2, col_p3 = st.columns([2,1,1])
   with col_p1:
       patient_name = st.text_input("Patient Name", placeholder="Enter patient name", key="patient_name")
   with col_p2:
       patient_age = st.number_input("Age", min_value=0, max_value=120, step=1, value=0, key="patient_age")
   with col_p3:
       patient_id = st.text_input("Patient ID", placeholder="Optional", key="patient_id")
 
   uploaded = st.file_uploader('Upload oral cavity image', type=['jpg','jpeg','png','bmp','tiff'])
   if uploaded is not None:
       col1, col2, col3 = st.columns([1,1,1])
       with col2:
           run_detection = st.button('Run AI Analysis', type='primary', use_container_width=True)
 
       if run_detection:
           if not patient_name:
               st.error("Please enter patient name before running analysis.")
               return
 
           # Save uploaded file to temp
           with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded.name.split('.')[-1]}") as tmp:
               tmp.write(uploaded.getbuffer())
               tmp_path = tmp.name
 
           try:
               with st.spinner("🔬 Analyzing image..."):
                   # Run inference
                   result = engine.predict_single_image(tmp_path, conf_threshold=DEFAULT_CONF, iou_threshold=DEFAULT_IOU)

                   # Filter detections: keep only highest confidence if more than one
                   detections = result.get('detections', []) or []
                   if len(detections) > 1:
                       best_det = max(detections, key=lambda d: d.get('confidence', 0))
                       result['detections'] = [best_det]

                   original_image = result.get('original_image', None)
                   if original_image is None:
                       st.error("Invalid image format")
                       return

                   annotated_image = draw_detections_rgb(original_image, result.get('detections', [])) if result.get('detections') else original_image

                   st.markdown("### Analysis Results")
                   col1, col2, col3 = st.columns([1,2,1])
                   with col2:
                       st.image(annotated_image, caption=f'AI Analysis - Patient: {patient_name}', width=400)

               # Heavy step: Prepare report + generate PDF
               with st.spinner("Preparing report for download..."):
                   # Build standardized report rows
                   rows = results_to_report_rows([result], include_patient_fields=True,
                                                 patient_name=patient_name, patient_age=patient_age, patient_id=patient_id)
                   df = pd.DataFrame(rows)

                   # Display dataframe with highlights
                   st.dataframe(df.style.apply(highlight_risk, axis=1), use_container_width=True, hide_index=True)

                   # PDF Download only
                   pdf_file_path = f"medical_report_{patient_name.replace(' ','_')}_{int(time.time())}.pdf"
                   generate_pdf_single(
                       patient_info={'name':patient_name,'age':patient_age,'id':patient_id},
                       report_df=df,
                       annotated_image=annotated_image,
                       file_path=pdf_file_path
                   )
                 
                   # Center the PDF download button
                   col_d1, col_d2, col_d3 = st.columns([1,1,1])
                   with col_d2:
                       with open(pdf_file_path,'rb') as f:
                           st.download_button(
                               label="Download Medical Report (PDF)",
                               data=f.read(),
                               file_name=pdf_file_path,
                               mime='application/pdf',
                               use_container_width=True
                           )

           finally:
               try: os.unlink(tmp_path)
               except Exception: pass
             
def show_batch_images(engine: YOLOInferenceEngine):
   st.markdown("### Multiple Images")
 
   # Upload multiple images
   files = st.file_uploader(
       'Select multiple oral cavity images',
       type=['jpg','jpeg','png','bmp','tiff'],
       accept_multiple_files=True,
       help="Select multiple oral cavity images for batch analysis"
   )
 
   # Show upload status
   if files and len(files) > 0:
       st.success(f" {len(files)} images selected for analysis")
       return files, None
   return None, None
 
def show_batch_zip(engine: YOLOInferenceEngine):
   st.markdown("### ZIP File")
 
   # Upload ZIP file
   zip_file = st.file_uploader(
       'Upload a ZIP file containing images',
       type=['zip'],
       help="Upload a ZIP file containing multiple oral cavity images"
   )
 
   # Show upload status
   if zip_file is not None:
       st.success("✅ ZIP file uploaded successfully")
       return None, zip_file
   return None, None
 
def show_batch(engine: YOLOInferenceEngine):
   st.markdown("### Batch Analysis")
 
   # Batch mode selection
   batch_mode = st.radio(
       "Select upload method:",
       ["Multiple Images", "ZIP File"],
       horizontal=True
   )
 
   if batch_mode == "Multiple Images":
       files, zip_file = show_batch_images(engine)
   else:
       files, zip_file = show_batch_zip(engine)
 
   # Centered Run button (similar to single image tab) — only show after upload
   run_batch_btn = False
   if (files and len(files) > 0) or (zip_file is not None):
       col_run_1, col_run_2, col_run_3 = st.columns([1,1,1])
       with col_run_2:
           run_batch_btn = st.button('Run AI Analysis', type='primary', use_container_width=True)
 
   # Initialize session storage for batch results
   if 'batch_results' not in st.session_state:
       st.session_state['batch_results'] = None
   if 'batch_elapsed' not in st.session_state:
       st.session_state['batch_elapsed'] = 0.0
 
   if run_batch_btn:
       results: List[Dict] = []
       start = time.time()
 
       if files:
           with tempfile.TemporaryDirectory() as tdir:
               paths = []
               for f in files:
                   p = os.path.join(tdir, f.name)
                   with open(p, 'wb') as out:
                       out.write(f.getbuffer())
                   paths.append(p)
               prog = st.progress(0)
               for i, p in enumerate(paths):
                   try:
                       res = engine.predict_single_image(
                           p,
                           conf_threshold=DEFAULT_CONF,
                           iou_threshold=DEFAULT_IOU
                       )
                   except Exception as e:
                       res = {
                           'image_path': p,
                           'error': str(e),
                           'detections': [],
                           'num_detections': 0
                       }
                   # Filter detections: keep only highest confidence if more than one
                   detections = res.get('detections', []) or []
                   if len(detections) > 1:
                       best_det = max(detections, key=lambda d: d.get('confidence', 0))
                       res['detections'] = [best_det]
                   results.append(res)
                   prog.progress((i + 1) / len(paths))
       elif zip_file is not None:
           with tempfile.TemporaryDirectory() as tdir:
               zip_path = os.path.join(tdir, 'upload.zip')
               with open(zip_path, 'wb') as out:
                   out.write(zip_file.getbuffer())
               with zipfile.ZipFile(zip_path, 'r') as z:
                   z.extractall(tdir)
               results = engine.predict_directory(
                   input_dir=tdir,
                   conf_threshold=DEFAULT_CONF,
                   iou_threshold=DEFAULT_IOU,
                   save_results=False
               )
               # Filter detections for each result
               for r in results:
                   detections = r.get('detections', []) or []
                   if len(detections) > 1:
                       best_det = max(detections, key=lambda d: d.get('confidence', 0))
                       r['detections'] = [best_det]
       else:
           st.warning('Please upload images or a ZIP file.')
           return
 
       elapsed = time.time() - start
       # Persist results for stable filtering UI across reruns
       st.session_state['batch_results'] = results
       st.session_state['batch_elapsed'] = elapsed
 
   # If we have persisted results, build report and show filters/table
   if st.session_state.get('batch_results'):
       results = st.session_state['batch_results']
 
       # Build standardized report rows across all results
       batch_rows = results_to_report_rows(results, include_patient_fields=False)
       df = pd.DataFrame(batch_rows)
 
       # Drop patient info for batch (if exists)
       df = df.drop(columns=[c for c in ['Patient Name','Age'] if c in df.columns])
 
       # ----------------- TOP SUMMARY (before filters) -----------------
       total_findings_all = len(df)
       classes_present = []
       if 'Class Name' in df.columns:
           classes_present = sorted([c for c in df['Class Name'].dropna().unique().tolist()])
     
       # Show overall metrics first
       elapsed = st.session_state.get('batch_elapsed', 0.0)
       col_m1, col_m2, col_m3 = st.columns(3)
       col_m1.metric('Images processed', len(results))
       col_m2.metric('Total findings (all)', total_findings_all)
       col_m3.metric('Time taken', f"{elapsed:.2f}s")
     
       # Show detected classes with better formatting
       if classes_present:
           st.markdown(f"**Detected classes:** {', '.join(classes_present)}")
       else:
           st.markdown("**Detected classes:** None")
     
       st.markdown("---")  # Separator line
 
       # ----------------- FILTERS -----------------
       st.markdown("### Filter Results")
       # Use fixed, clinically-meaningful options so users always see all choices.
       base_df = df.copy()
       col_f1, col_f2, col_f3 = st.columns(3)
 
       # Class filter (always show all classes)
       all_class_options = ["All"] + CLASS_NAMES
       with col_f1:
           selected_class = st.selectbox(
               "Class name",
               all_class_options,
               index=0,
               key="filter_class",
               help="Filter findings by predicted class"
           )
       if selected_class != "All" and "Class Name" in base_df.columns:
           if selected_class not in base_df['Class Name'].unique().tolist():
               st.info(f"No findings for class: {selected_class} in the uploaded images.")
 
       # Risk filter (always show the full risk set)
       all_risk_options = ["All"] + list(risk_color_mapping.keys())
       with col_f2:
           selected_risk = st.selectbox(
               "Risk level",
               all_risk_options,
               index=0,
               key="filter_risk",
               help="Filter by clinical risk level"
           )
       if selected_risk != "All" and "Risk Assessment" in base_df.columns:
           if selected_risk not in base_df['Risk Assessment'].unique().tolist():
               st.info(f"No findings for risk level: {selected_risk} in the uploaded images.")
 
       # Confidence filter (range slider 0.00–1.00) - default to show ALL results
       with col_f3:
           conf_min_val, conf_max_val = st.slider(
               "Confidence range",
               min_value=0.0,
               max_value=1.0,
               value=(0.0, 1.0),  # Default to show ALL results
               step=0.01,
               key="filter_conf_range",
               help="Only show findings within this confidence range"
           )
 
       # ----------------- APPLY FILTERS -----------------
       if selected_class != "All" and "Class Name" in df.columns:
           df = df[df['Class Name'] == selected_class]
 
       if selected_risk != "All" and "Risk Assessment" in df.columns:
           df = df[df['Risk Assessment'] == selected_risk]
 
       # Only apply confidence filter if it's not the default range (0.0 to 1.0)
       if "Confidence" in df.columns and (conf_min_val != 0.0 or conf_max_val != 1.0):
           df = df[(df["Confidence"] >= conf_min_val) & (df["Confidence"] <= conf_max_val)]
 
       if df.empty:
           st.info("No records match the current filters. Try selecting 'All' for class and risk, or widen the confidence range.")
           return
 
       # ----------------- FILTERED RESULTS SUMMARY -----------------
       col_f1, col_f2, col_f3 = st.columns(3)
       col_f1.metric('Findings shown (after filters)', len(df))
       col_f2.metric('Currently displayed', f"{len(df)} of {total_findings_all}")
       col_f3.metric('Filter efficiency', f"{(len(df)/total_findings_all*100):.1f}%" if total_findings_all > 0 else "0%")
 
       # ----------------- DISPLAY WITH COLOR -----------------
       st.dataframe(
           df.style.apply(highlight_risk, axis=1),
           use_container_width=True,
           hide_index=True
       )
 
       # ----------------- CSV AND PDF DOWNLOAD -----------------
       batch_csv_path = f"yolo_batch_report_{int(time.time())}.csv"
       df.to_csv(batch_csv_path, index=False)

       batch_pdf_path = f"yolo_batch_report_{int(time.time())}.pdf"
       generate_pdf_batch(df, batch_pdf_path)

       col_d1, col_d2, col_d3 = st.columns([1,1,1])
       with col_d2:
           with open(batch_csv_path, 'rb') as f:
               st.download_button(
                   "Download Batch Report (CSV)",
                   data=f.read(),
                   file_name=batch_csv_path,
                   mime='text/csv',
                   use_container_width=True
               )
           with open(batch_pdf_path, 'rb') as f:
               st.download_button(
                   "Download Batch Report (PDF)",
                   data=f.read(),
                   file_name=batch_pdf_path,
                   mime='application/pdf',
                   use_container_width=True
               )

       # --------- Single finding PDF (with annotated image) for selected row ---------
       st.markdown("---")
       st.markdown("#### Download Detailed Report for a Single Finding")
       # Prompt user to enter S.No for detailed PDF report
       min_sno = 1
       max_sno = len(df)
       entered_sno = st.number_input(
           "Enter the S.No for detailed PDF report",
           min_value=min_sno,
           max_value=max_sno,
           value=min_sno,
           step=1,
           key="batch_select_sno"
       )
       # Only proceed if there are rows
       if len(df) > 0 and entered_sno is not None:
           # Find the row with S.No == entered_sno
           selected_rows = df[df['S.No'] == entered_sno]
           if not selected_rows.empty:
               selected_row = selected_rows.iloc[0]
               # Find the corresponding result dictionary from st.session_state['batch_results']
               results = st.session_state.get('batch_results', [])
               selected_sno = selected_row.get('S.No', None)
               matched_result = None
               # Find the result whose detections match this finding (by order)
               finding_counter = 1
               for r in results:
                   if 'error' in r and r['error']:
                       if finding_counter == selected_sno:
                           matched_result = r
                           break
                       finding_counter += 1
                       continue
                   detections = r.get('detections', []) or []
                   if len(detections) == 0:
                       if finding_counter == selected_sno:
                           matched_result = r
                           break
                       finding_counter += 1
                       continue
                   for det in detections:
                       if finding_counter == selected_sno:
                           matched_result = {
                               'original_image': r.get('original_image'),
                               'detections': [det],
                               'image_path': r.get('image_path', '')
                           }
                           break
                       finding_counter += 1
                   if matched_result:
                       break
               # Prepare annotated image as in single mode
               annotated_image = None
               if matched_result is not None:
                   original_image = matched_result.get('original_image')
                   detections = matched_result.get('detections', [])
                   if original_image is not None and detections:
                       annotated_image = draw_detections_rgb(original_image, detections)
                   elif original_image is not None:
                       annotated_image = original_image
               # Prepare patient info (minimal in batch mode)
               patient_info = {
                   'name': '',
                   'age': '',
                   'id': selected_row.get('Patient ID', '')
               }
               # Remove the "Select" column if present
               report_df_single = selected_rows.copy()
               if "Select" in report_df_single.columns:
                   report_df_single = report_df_single.drop(columns=["Select"])
               # Generate PDF for the single row
               single_pdf_path = f"single_finding_report_{selected_row.get('Patient ID','')}_{selected_row.get('S.No','')}_{int(time.time())}.pdf"
               generate_pdf_single(
                   patient_info=patient_info,
                   report_df=report_df_single,
                   annotated_image=annotated_image,
                   file_path=single_pdf_path
               )
               # Download button for detailed PDF
               with open(single_pdf_path, 'rb') as f:
                   st.download_button(
                       "Download Detailed Report (PDF) for Selected Finding",
                       data=f.read(),
                       file_name=single_pdf_path,
                       mime='application/pdf',
                       use_container_width=True
                   )
   else:
       pass

 
def main():
    # Professional sidebar navigation
    with st.sidebar:
        st.markdown("### Clinical Analysis")

        # Navigation options
        page = st.radio(
            "Analysis Mode:",
            ["Single Image", "Batch Analysis", "Live Mode"],
            index=0
        )

        st.markdown("---")
        st.markdown("### Classification Guide")
        st.markdown("""
        **Healthy:** Normal oral tissue  
        **Benign:** Non-cancerous lesions  
        **OPMD:** Potentially malignant disorders  
        **OCA:** Oral cavity carcinoma
        """)

        st.markdown("---")
        st.markdown("### Clinical Notice")
        st.caption("This tool assists clinical screening. Professional diagnosis required.")

    # Main content area
    try:
        engine = load_engine()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Display selected page
    if page == "Single Image":
        show_single_image(engine)
    elif page == "Batch Analysis":
        show_batch(engine)
    elif page == "Live Mode":
        show_live_mode(engine)

if __name__ == '__main__':
    main()
 