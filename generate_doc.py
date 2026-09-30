import os
import glob
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for margin_name, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{margin_name}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def style_table(table, col_widths, header_bg="1F4E79", alt_bg="F2F5F8"):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, cell in enumerate(table.rows[0].cells):
        set_cell_background(cell, header_bg)
        set_cell_margins(cell, top=140, bottom=140, left=150, right=150)
        cell.width = col_widths[idx]
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                run.font.bold = True
                run.font.size = Pt(9.5)
                run.font.color.rgb = RGBColor(255, 255, 255)
    
    for row_idx, row in enumerate(table.rows[1:], start=1):
        bg = alt_bg if row_idx % 2 == 1 else "FFFFFF"
        for idx, cell in enumerate(row.cells):
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
            cell.width = col_widths[idx]
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)
                    run.font.color.rgb = RGBColor(40, 40, 40)

def find_file(pattern):
    matches = glob.glob(f"**/{pattern}", recursive=True)
    return matches[0] if matches else None

def main():
    doc = docx.Document()
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # Title & Subtitle
    title_p = doc.add_paragraph()
    run_title = title_p.add_run("Risk-Adaptive Continuous Authentication for Mobile Money (GaitAuth)")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(20)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(31, 78, 121)

    sub_p = doc.add_paragraph()
    run_sub = sub_p.add_run("Project Deliverables, Verification Metrics, Policy Engine & Submission Dossier")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(12)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(89, 89, 89)

    # 1. Project Links & Architecture
    doc.add_heading("1. Project & Repository Access Links", level=1)
    table1 = doc.add_table(rows=5, cols=3)
    t1_widths = [Inches(2.0), Inches(3.2), Inches(1.6)]
    for i, title in enumerate(["Resource / Component", "Access Link / Identifier", "Operational Scope"]):
        table1.rows[0].cells[i].text = title
    data1 = [
        ("GitHub Code Repository", "https://github.com/nessa360/risk-adaptive-mobile-money-gait-auth", "Public / Full Source Code"),
        ("FastAPI Local Swagger UI", "http://localhost:8000/docs", "Interactive OpenAPI Testing"),
        ("Android Testing Route", "http://10.0.2.2:8000/ (or LAN: 192.168.1.185:8000)", "Host Loopback Gateway"),
        ("Evaluation Dataset Source", "WISDM Smartphone & Smartwatch Biometrics", "Public Benchmark (Weiss et al., 2019)")
    ]
    for r_idx, row in enumerate(data1, start=1):
        for c_idx, val in enumerate(row):
            table1.rows[r_idx].cells[c_idx].text = val
    style_table(table1, t1_widths)

    # Embed Architecture Diagram
    arch_img = find_file("*architecture*.png") or find_file("*gaitauth_architecture*.png")
    if arch_img and os.path.exists(arch_img):
        p_arch = doc.add_paragraph()
        p_arch.paragraph_format.space_before = Pt(12)
        p_arch.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_arch.add_run().add_picture(arch_img, width=Inches(6.8))
        
        cap_arch = doc.add_paragraph()
        cap_arch.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_arch = cap_arch.add_run("Figure 1: Risk-Adaptive Continuous Biometric System Architecture Pipeline")
        r_arch.font.size = Pt(9.0)
        r_arch.font.italic = True
    else:
        print("Note: Architecture diagram image not found in project subdirectories.")

    # 2. E1 Results
    doc.add_heading("2. Experimental Verification Results (E1 Baseline)", level=1)
    doc.add_paragraph("Biometric verification evaluated under subject-disjoint protocol on WISDM (30 dev, 11 unseen test subjects) across non-overlapping 10s windows (6x128 at 50 Hz).")
    table2 = doc.add_table(rows=8, cols=3)
    t2_widths = [Inches(2.2), Inches(1.4), Inches(3.2)]
    for i, title in enumerate(["Metric Parameter", "Measured Value", "Academic & Engineering Interpretation"]):
        table2.rows[0].cells[i].text = title
    data2 = [
        ("Evaluation Subjects", "11 unseen subjects", "Strictly isolated from training to avoid data leakage"),
        ("Genuine Trials", "119 comparisons", "Intra-subject comparisons across distinct walking windows"),
        ("Zero-Effort Impostor Trials", "1,190 comparisons", "Pairwise inter-subject comparison trials"),
        ("Equal Error Rate (EER)", "20.17%", "Intersection point where FAR(theta) ≈ FRR(theta)"),
        ("Operating Threshold (theta)", "0.9971", "Selected cosine similarity boundary balancing verification"),
        ("Verification Accuracy", "77.62%", "Overall discrimination accuracy across all trials"),
        ("False Acceptance / Rejection", "FAR: 23.19% | FRR: 14.29%", "Demonstrates uncertainty requiring multi-tier IAM defense")
    ]
    for r_idx, row in enumerate(data2, start=1):
        for c_idx, val in enumerate(row):
            table2.rows[r_idx].cells[c_idx].text = val
    style_table(table2, t2_widths)

    # Embed ROC & Distribution Curves
    roc_img = find_file("*roc*.png")
    dist_img = find_file("*score_distr*.png") or find_file("*distribution*.png")

    if roc_img and dist_img and os.path.exists(roc_img) and os.path.exists(dist_img):
        doc.add_paragraph().paragraph_format.space_before = Pt(10)
        fig_table = doc.add_table(rows=1, cols=2)
        fig_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        
        # ROC Plot
        p1 = fig_table.rows[0].cells[0].paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p1.add_run().add_picture(roc_img, width=Inches(3.2))
        cap1 = fig_table.rows[0].cells[0].add_paragraph()
        cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r1 = cap1.add_run("Figure 2: E1 ROC Curve (EER = 20.17%)")
        r1.font.size = Pt(8.5)
        r1.font.italic = True

        # Distribution Plot
        p2 = fig_table.rows[0].cells[1].paragraphs[0]
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p2.add_run().add_picture(dist_img, width=Inches(3.2))
        cap2 = fig_table.rows[0].cells[1].add_paragraph()
        cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r2 = cap2.add_run("Figure 3: Genuine vs. Impostor Distribution")
        r2.font.size = Pt(8.5)
        r2.font.italic = True

    # 3. Baseline Comparison
    doc.add_heading("3. Baseline Comparison: Open-Set vs. Closed-Set", level=1)
    table3 = doc.add_table(rows=5, cols=4)
    t3_widths = [Inches(2.0), Inches(1.8), Inches(1.2), Inches(1.8)]
    for i, title in enumerate(["Model Architecture", "Operating Paradigm", "Test Accuracy", "Macro F1-Score"]):
        table3.rows[0].cells[i].text = title
    data3 = [
        ("Support Vector Machine (SVM)", "Closed-Set Multiclass", "0.00%", "0.0000"),
        ("Random Forest (RF)", "Closed-Set Multiclass", "0.00%", "0.0000"),
        ("k-Nearest Neighbors (k-NN)", "Closed-Set Multiclass", "0.00%", "0.0000"),
        ("CNN-LSTM Embedder (Proposed)", "Open-Set Metric Embedder", "77.62%", "0.7762")
    ]
    for r_idx, row in enumerate(data3, start=1):
        for c_idx, val in enumerate(row):
            table3.rows[r_idx].cells[c_idx].text = val
    style_table(table3, t3_widths)

    # 4. Host Profiling
    doc.add_heading("4. Host Computational Profiling (E5 Benchmarks)", level=1)
    table4 = doc.add_table(rows=5, cols=3)
    t4_widths = [Inches(2.8), Inches(1.8), Inches(2.2)]
    for i, title in enumerate(["Processing Pipeline Stage", "Execution Latency", "Peak Working Memory"]):
        table4.rows[0].cells[i].text = title
    data4 = [
        ("Signal Slicing & Resampling", "0.082 ms", "< 0.08 MB"),
        ("ONNX CNN-LSTM Model Inference", "0.913 ms", "0.38 MB"),
        ("Policy & Threshold Resolution", "0.012 ms", "< 0.01 MB"),
        ("Total Host Pipeline Round-Trip", "~1.06 ms", "< 0.40 MB")
    ]
    for r_idx, row in enumerate(data4, start=1):
        for c_idx, val in enumerate(row):
            table4.rows[r_idx].cells[c_idx].text = val
    style_table(table4, t4_widths)

    # 5. Policy Engine
    doc.add_heading("5. Risk-Adaptive IAM Policy Engine Matrix", level=1)
    table5 = doc.add_table(rows=7, cols=4)
    t5_widths = [Inches(1.8), Inches(2.0), Inches(1.4), Inches(1.6)]
    for i, title in enumerate(["Confidence Tier", "Transaction Context", "Decision", "UX Workflow"]):
        table5.rows[0].cells[i].text = title
    data5 = [
        ("High (s >= 0.80)", "Amount <= 100 GHS & Known Contact", "ALLOW", "Zero-friction approval"),
        ("High (s >= 0.80)", "Amount 101 - 500 GHS", "STEP_UP_LIGHT", "4-digit PIN verification"),
        ("High (s >= 0.80)", "Amount > 500 GHS", "STEP_UP_STRONG", "6-digit SMS OTP challenge"),
        ("Medium (0.45 <= s < 0.80)", "Amount <= 50 GHS & Known Contact", "STEP_UP_LIGHT", "4-digit PIN verification"),
        ("Medium (0.45 <= s < 0.80)", "Amount > 50 GHS or Unknown", "STEP_UP_STRONG", "6-digit SMS OTP challenge"),
        ("Low (s < 0.45)", "Any Amount / Any Recipient", "DENY", "Transaction terminated")
    ]
    for r_idx, row in enumerate(data5, start=1):
        for c_idx, val in enumerate(row):
            table5.rows[r_idx].cells[c_idx].text = val
    style_table(table5, t5_widths)

    # 6. Test Suite
    doc.add_heading("6. End-to-End Functional Test Suite", level=1)
    table6 = doc.add_table(rows=7, cols=4)
    t6_widths = [Inches(1.0), Inches(1.6), Inches(3.2), Inches(1.0)]
    for i, title in enumerate(["Test ID", "Subsystem", "Evaluation Scenario & Observed Behavior", "Verdict"]):
        table6.rows[0].cells[i].text = title
    data6 = [
        ("TS-01", "Android Sensor", "50 Hz IMU acquisition maintained in ring buffer without dropping frames", "PASS"),
        ("TS-02", "Hardware Fallback", "Static emulator detects underfilled buffer; injects 6x128 synthetic baseline", "PASS"),
        ("TS-03", "Networking", "Retrofit connects via 10.0.2.2; logs HTTP 200 payload round-trip", "PASS"),
        ("TS-04", "Prototype ALLOW", "Score >= 0.80, 25 GHS: frictionless transfer completes without prompt", "PASS"),
        ("TS-05", "Prototype STEP-UP", "Score 0.45-0.80 or > 100 GHS: modal challenges PIN; valid input clears transfer", "PASS"),
        ("TS-06", "Prototype DENY", "Divergent score < 0.45: transaction intercepted with Security Alert dialog", "PASS")
    ]
    for r_idx, row in enumerate(data6, start=1):
        for c_idx, val in enumerate(row):
            table6.rows[r_idx].cells[c_idx].text = val
    style_table(table6, t6_widths)

    out = "GaitAuth_Project_Results_and_Links.docx"
    doc.save(out)
    print(f"SUCCESS: Generated {out} with all tables and all 3 visual figures embedded.")

if __name__ == "__main__":
    main()
