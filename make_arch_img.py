import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.axis('off')

# Styling helper
def draw_box(x, y, w, h, title, subtitle, bg_color='#FFFFFF', border_color='#1F4E79', dashed=False):
    ls = '--' if dashed else '-'
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.12",
        ec=border_color, fc=bg_color, lw=1.5, ls=ls
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.65, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1A1A1A')
    if subtitle:
        ax.text(x + w/2, y + h*0.32, subtitle, ha='center', va='center', fontsize=8, color='#555555')

def draw_arrow(x1, y1, x2, y2, color='#1F4E79', label=''):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(facecolor=color, edgecolor=color, width=1.5, headwidth=7, shrink=0.05)
    )
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.18, label, ha='center', va='bottom', fontsize=8, fontweight='bold', color=color)

# Subsystem Containers
c1 = patches.FancyBboxPatch((0.5, 0.8), 4.3, 8.8, boxstyle="round,pad=0.2", ec='#1F4E79', fc='#F8FAFC', lw=2)
c2 = patches.FancyBboxPatch((5.3, 0.8), 4.7, 8.8, boxstyle="round,pad=0.2", ec='#0369A1', fc='#F8FAFC', lw=2)
c3 = patches.FancyBboxPatch((10.5, 0.8), 4.0, 8.8, boxstyle="round,pad=0.2", ec='#6B21A8', fc='#FAF5FF', lw=2)
ax.add_patch(c1); ax.add_patch(c2); ax.add_patch(c3)

ax.text(2.65, 9.25, "ANDROID CLIENT SUBSYSTEM", ha='center', fontsize=11, fontweight='bold', color='#1F4E79')
ax.text(7.65, 9.25, "FASTAPI INFERENCE & IAM ENGINE", ha='center', fontsize=11, fontweight='bold', color='#0369A1')
ax.text(12.5, 9.25, "ENROLLMENT & TEMPLATE STORE", ha='center', fontsize=11, fontweight='bold', color='#6B21A8')

# 1. Android Client Nodes
draw_box(0.9, 8.0, 3.5, 0.8, "Smartphone IMU Hardware", "Tri-axial Accel & Gyroscope (50 Hz)")
draw_box(0.9, 6.7, 3.5, 0.8, "GaitSensorService", "Unthrottled Foreground Service (specialUse)", bg_color='#EFF6FF')
draw_box(0.9, 5.4, 3.5, 0.8, "GaitDataBuffer", "Circular Ring Buffer: [6 × 128] Samples")
draw_box(0.9, 4.2, 3.5, 0.7, "Stationary Baseline Fallback", "DEBUG ONLY: Calibrated Synthetic Vector", bg_color='#FEF3C7', border_color='#D97706', dashed=True)
draw_box(0.9, 2.9, 3.5, 0.8, "MainActivity UI & Dispatcher", "Retrofit 2 / OkHttp HTTP Client Layer")
draw_box(0.9, 1.2, 3.5, 1.2, "Adaptive UI Resolution", "ALLOW (Frictionless) | STEP_UP (PIN/OTP) | DENY", bg_color='#F0FDF4', border_color='#16A34A')

draw_arrow(2.65, 8.0, 2.65, 7.5)
draw_arrow(2.65, 6.7, 2.65, 6.2)
draw_arrow(2.65, 5.4, 2.65, 4.9)
draw_arrow(2.65, 4.2, 2.65, 3.7)

# 2. FastAPI Backend Nodes
draw_box(5.8, 8.0, 3.7, 0.8, "POST /evaluate Endpoint", "FastAPI Router & Pydantic Validation")
draw_box(5.8, 6.7, 3.7, 0.8, "Sliding-Window Normalizer", "Matrix Slicing & Reshaping: [1, 6, 128]", bg_color='#F0F9FF')
draw_box(5.8, 5.4, 3.7, 0.8, "ONNX Runtime Engine", "CNN-LSTM Embedder: e_live in R^128", bg_color='#DBEAFE', border_color='#1D4ED8')
draw_box(5.8, 4.1, 3.7, 0.8, "Open-Set Metric Evaluator", "Cosine Similarity: s = cos(e_live, e_enrolled)")
draw_box(5.8, 2.6, 3.7, 1.1, "Risk-Adaptive IAM Policy", "High (>=0.80) | Med (0.45-0.80) | Low (<0.45)", bg_color='#FFEDD5', border_color='#EA580C')
draw_box(5.8, 1.2, 3.7, 0.8, "IAM Decision Dispatcher", "{ ALLOW | STEP_UP_LIGHT | STEP_UP_STRONG | DENY }")

draw_arrow(7.65, 8.0, 7.65, 7.5)
draw_arrow(7.65, 6.7, 7.65, 6.2)
draw_arrow(7.65, 5.4, 7.65, 4.9)
draw_arrow(7.65, 4.1, 7.65, 3.7)
draw_arrow(7.65, 2.6, 7.65, 2.0)

# 3. Enrollment Store Nodes
draw_box(11.0, 6.7, 3.0, 1.1, "Enrollment Module", "Multi-Session Walking Capture\nQuality Check & Mean Vector", bg_color='#FFFFFF', border_color='#6B21A8')
draw_box(11.0, 4.1, 3.0, 1.1, "Template Store", "Enrolled Baseline: e_enrolled\nStored Reference Embedding", bg_color='#FFFFFF', border_color='#6B21A8')

draw_arrow(12.5, 6.7, 12.5, 5.2)
draw_arrow(11.0, 4.65, 9.5, 4.5, color='#6B21A8', label='e_enrolled')

# Inter-Subsystem Corridors
# Client -> Server
ax.annotate('', xy=(5.8, 8.4), xytext=(4.4, 3.3),
            arrowprops=dict(arrowstyle="->", color='#1F4E79', lw=2, connectionstyle="angle,angleA=0,angleB=90,rad=8"))
ax.text(5.1, 7.2, "HTTP POST\n[6, 128]", ha='center', fontsize=8, fontweight='bold', color='#1F4E79')

# Server -> Client
ax.annotate('', xy=(4.4, 1.8), xytext=(5.8, 1.6),
            arrowprops=dict(arrowstyle="->", color='#16A34A', lw=2, connectionstyle="arc3,rad=-0.1"))
ax.text(5.1, 1.9, "HTTP 200 OK\nDecision", ha='center', fontsize=8, fontweight='bold', color='#16A34A')

# Regulatory Footer
footer = patches.FancyBboxPatch((0.5, 0.1), 14.0, 0.5, boxstyle="round,pad=0.08", ec='#D97706', fc='#FFFBEB', lw=1)
ax.add_patch(footer)
ax.text(7.5, 0.35, "Governance & Compliance: Compliant with Ghana Data Protection Act, 2012 (Act 843) | Continuous Adaptive Friction IAM",
        ha='center', va='center', fontsize=7.5, fontweight='bold', color='#92400E')

plt.tight_layout()
out_path = "results/figures/system_architecture.png"
plt.savefig(out_path, bbox_inches='tight', dpi=300)
print(f"SUCCESS: Generated {out_path}")
