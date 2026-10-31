"""
Generate Figure 1: System Architecture Diagram for CRISPR Guide AI
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, "publication_package", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Title
ax.text(50, 97, "CRISPR Guide AI: End-to-End System Architecture", ha="center", va="center", fontsize=15, fontweight="bold", color="#1a252f")
ax.text(50, 94, "Dual-Task Prediction Engine, Feature Pipeline, Biophysical Rule Engine, and Web Application", ha="center", va="center", fontsize=10, style="italic", color="#555")

def draw_box(x, y, w, h, title, lines, color_border, color_fill, title_color="#000"):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.2", linewidth=1.5, edgecolor=color_border, facecolor=color_fill)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 3, title, ha="center", va="center", fontsize=10, fontweight="bold", color=title_color)
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h - 7 - i * 3.5, line, ha="center", va="center", fontsize=8.5, color="#2c3e50")

# 1. Input Module
draw_box(5, 75, 40, 14, "1. User Input & Validation", [
    "Input: 20-nt Protospacer or 23-nt Target Sequence",
    "Biological Validation: ACGT verification, Length checks",
    "Canonical PAM Extension: Appends 'TGG' (explicitly flagged)"
], "#2980b9", "#ebf5fb", "#1b4f72")

# 2. 600-Feature Extraction
draw_box(55, 75, 40, 14, "2. 600-Feature Biophysical Pipeline", [
    "28 Compositional & Transcriptional Features",
    "444 Positional 1-mers (92) & 2-mers (352)",
    "80 Global k-mers & 48 n-Gapped Dinucleotides",
    "StandardScaler (Fitted on 80% Development Pool Only)"
], "#27ae60", "#eafaf1", "#145a32")

# Arrow from Input to Features
ax.annotate("", xy=(54, 82), xytext=(46, 82), arrowprops=dict(arrowstyle="->", lw=2, color="#2c3e50"))

# Arrow from Features to Model Hub
ax.annotate("", xy=(50, 68), xytext=(50, 74), arrowprops=dict(arrowstyle="->", lw=2, color="#2c3e50"))

# 3. Model Engine (Dual-Task)
rect_mid = patches.FancyBboxPatch((5, 34), 90, 33, boxstyle="round,pad=1.2", linewidth=1.5, edgecolor="#8e44ad", facecolor="#f5eef8")
ax.add_patch(rect_mid)
ax.text(50, 64, "3. Cell-Line Specific Dual-Task Predictive Engine (HeLa, HCT116, HEK293T, HL60)", ha="center", va="center", fontsize=11, fontweight="bold", color="#512e5f")

# Sub-box Regression
draw_box(8, 37, 40, 24, "Continuous Efficacy Regression", [
    "Target: Inverted Depletion Efficiency [0, 1]",
    "Models: Ridge, RF, XGBoost, SVR",
    "Weighted Ensemble (Convex OOF Simplex)",
    "Output: Relative Efficacy Score (0 - 100)",
    "Empirical Uncertainty: RF Tree Dispersion (±1.96σ)",
    "Baseline: DummyRegressor (Mean/Median)"
], "#2980b9", "#ffffff", "#154360")

# Sub-box Classification
draw_box(52, 37, 40, 24, "Binary Activity Classification", [
    "Target: Active Knockout (log2FC <= -1.0)",
    "Models: Logistic, RF, XGBoost, SVC, Ridge Clf",
    "Operating Cutoff: Development Youden's J",
    "Output: Active / Inactive Label & Calibrated Prob",
    "Specificity: Strict TN / (TN + FP)",
    "Baseline: DummyClassifier (Most Frequent)"
], "#c0392b", "#ffffff", "#78281f")

# Arrow down to Post-Processing
ax.annotate("", xy=(50, 26), xytext=(50, 33), arrowprops=dict(arrowstyle="->", lw=2, color="#2c3e50"))

# 4. Post-processing & Biophysical Analysis
draw_box(5, 12, 42, 13, "4. Biophysical Rules & Explainability", [
    "GC Content Checking (40% - 60% optimal window)",
    "Poly-T Termination Signals (TTTT motif flags)",
    "Positional Feature Saliency (Seed & Cleavage Flanks)"
], "#d35400", "#fef5e7", "#7e3402")

# 5. Full Stack Application Delivery
draw_box(53, 12, 42, 13, "5. Full-Stack Software Interface", [
    "FastAPI Asynchronous REST API Backend",
    "Single & Batch High-Throughput Prediction (1860 seq/s)",
    "React / TypeScript / Tailwind CSS Interactive UI"
], "#16a085", "#e8f8f5", "#0e6252")

# Connection between 4 and 5
ax.annotate("", xy=(52, 18.5), xytext=(48, 18.5), arrowprops=dict(arrowstyle="<->", lw=1.5, color="#2c3e50"))

plt.tight_layout()
fig1_png = os.path.join(FIG_DIR, "Fig1_System_Architecture.png")
fig1_svg = os.path.join(FIG_DIR, "Fig1_System_Architecture.svg")
fig.savefig(fig1_png, dpi=300)
fig.savefig(fig1_svg)
plt.close(fig)
print(f"Saved Figure 1 to: {fig1_png}")
