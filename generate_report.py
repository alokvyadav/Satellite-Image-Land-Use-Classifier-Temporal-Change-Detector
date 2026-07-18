import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Configurations
WORKSPACE = "C:/Users/alok1/.gemini/antigravity-ide/scratch/satellite_change_detection"
RESULTS_DIR = os.path.join(WORKSPACE, "results")
CHECKPOINT_DIR = os.path.join(WORKSPACE, "checkpoints")
PDF_PATH = os.path.join(WORKSPACE, "Satellite_Change_Detection_Report.pdf")

def read_file_content(path, default="Data not available."):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return default

def get_image(path, width, height):
    if os.path.exists(path):
        try:
            return RLImage(path, width=width, height=height)
        except Exception as e:
            return Paragraph(f"Error loading image: {os.path.basename(path)}", getSampleStyleSheet()['Normal'])
    return Paragraph(f"Image not found: {os.path.basename(path)}", getSampleStyleSheet()['Normal'])

def main():
    print("Compiling PDF report...")
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles for premium styling
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1f4068'),
        alignment=1, # Center
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#1f4068'),
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=8
    )
    
    code_style = ParagraphStyle(
        'DocCode',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=6
    )
    
    table_text_style = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#2c3e50')
    )
    
    story = []
    
    # ----------------------------------------------------
    # PAGE 1: TITLE & EXECUTIVE SUMMARY
    # ----------------------------------------------------
    story.append(Spacer(1, 40))
    story.append(Paragraph("Satellite Image Land-Use Classifier &<br/>Temporal Change Detector", title_style))
    story.append(Paragraph("A Deep Learning and Embedding Cosine Similarity System", ParagraphStyle('Sub', parent=styles['Normal'], alignment=1, fontSize=12, spaceAfter=30)))
    
    story.append(Paragraph("1. Executive Summary", h1_style))
    summary_text = (
        "This project develops an end-to-end computer vision system to classify land-use categories from high-resolution "
        "satellite tiles and monitor geographic change over time. The primary training and validation dataset is "
        "EuroSAT (27,000 Sentinel-2 tiles representing 10 land cover classes), split geographically using a spatial block split "
        "to prevent spatial data leakage. The UC Merced Land Use dataset (2,100 aerial images, 21 classes) is used as a "
        "cross-dataset holdout set, evaluated by mapping its 21 labels semantically onto the 10 EuroSAT classes. "
        "Our system compares a scratch-trained 3-layer CNN against a transfer-learned ResNet-18 model utilizing a two-phase "
        "fine-tuning strategy. Temporal change detection is implemented via cosine similarity of ResNet-18 feature embeddings, "
        "demonstrating excellent performance across three key operating points. The system is deployed via an interactive "
        "Streamlit dashboard containing GradCAM spatial saliency heatmaps."
    )
    story.append(Paragraph(summary_text, body_style))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("System Architecture Overview:", ParagraphStyle('SubH', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, spaceAfter=8)))
    
    # Simple table outlining specs
    data_specs = [
        [Paragraph("<b>Component</b>", table_text_style), Paragraph("<b>Specification</b>", table_text_style)],
        [Paragraph("Primary Dataset", table_text_style), Paragraph("EuroSAT RGB (27,000 tiles, 10 classes)", table_text_style)],
        [Paragraph("Holdout Dataset", table_text_style), Paragraph("UC Merced Land Use (2,100 tiles, 21 classes)", table_text_style)],
        [Paragraph("Base Model Architecture", table_text_style), Paragraph("ResNet-18 (Pretrained on ImageNet)", table_text_style)],
        [Paragraph("Training Method", table_text_style), Paragraph("Two-phase transfer learning (frozen vs unfreeze blocks 3-4)", table_text_style)],
        [Paragraph("Change Detector", table_text_style), Paragraph("Embedding-based Cosine Similarity (512-dim)", table_text_style)],
        [Paragraph("Dashboard Features", table_text_style), Paragraph("Streamlit app + GradCAM overlays + multi-threshold toggles", table_text_style)]
    ]
    t_specs = Table(data_specs, colWidths=[180, 320])
    t_specs.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f2f4f7')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d0d5dd')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_specs)
    story.append(PageBreak())
    
    # ----------------------------------------------------
    # PAGE 2: DATA PIPELINE & CLASS DISTRIBUTIONS
    # ----------------------------------------------------
    story.append(Paragraph("2. Data Pipeline & Spatial Block Splitting", h1_style))
    pipeline_text = (
        "In satellite image classification, adjacent tiles are highly spatially correlated. Spliting the dataset "
        "randomly creates 'spatial data leakage', where highly similar neighboring tiles end up in both training "
        "and validation splits, artificially inflating evaluation accuracy. To evaluate generalization accurately, "
        "we implement a <b>Spatial Block Split</b> based on the official TorchGeo longitude-based partition, dividing the "
        "27,000 EuroSAT images into Train (16,200 tiles, 60%), Val (5,400 tiles, 20%), and Test (5,400 tiles, 20%) splits. "
        "The class distribution is balanced across splits, as shown below."
    )
    story.append(Paragraph(pipeline_text, body_style))
    
    # Embed class distribution plot
    story.append(Spacer(1, 10))
    story.append(get_image(os.path.join(RESULTS_DIR, "class_distribution.png"), 480, 240))
    
    # Embed dataset sample grid description
    story.append(Spacer(1, 10))
    story.append(Paragraph("Visualized Class Samples (Forest, River, Industrial, Crops, etc.):", ParagraphStyle('SubH2', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, spaceAfter=8)))
    story.append(get_image(os.path.join(RESULTS_DIR, "eurosat_class_samples.png"), 480, 160))
    story.append(PageBreak())
    
    # ----------------------------------------------------
    # PAGE 3: BASELINE CNN PERFORMANCE
    # ----------------------------------------------------
    story.append(Paragraph("3. Baseline CNN Model Performance", h1_style))
    baseline_intro = (
        "We train a 3-layer convolutional neural network (Conv $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool $\rightarrow$ Dropout) "
        "from scratch as a performance floor. The model was trained on 64x64 inputs for 10 epochs using Adam "
        "optimizer. The training and validation loss curves show convergence without severe overfitting, though overall "
        "representation capacity remains limited compared to transfer learning."
    )
    story.append(Paragraph(baseline_intro, body_style))
    
    # Embed Baseline Loss curves
    story.append(Spacer(1, 10))
    story.append(get_image(os.path.join(RESULTS_DIR, "baseline_loss_curves.png"), 320, 200))
    
    # Report per-class F1 for Baseline
    story.append(Spacer(1, 15))
    story.append(Paragraph("Baseline CNN Validation Classification Report:", ParagraphStyle('SubH3', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, spaceAfter=6)))
    report_text = read_file_content(os.path.join(RESULTS_DIR, "baseline_classification_report.txt"))
    for line in report_text.splitlines():
        story.append(Paragraph(line.replace(" ", "&nbsp;"), code_style))
        
    story.append(PageBreak())
    
    # ----------------------------------------------------
    # PAGE 4: TRANSFER LEARNING & ABLATION
    # ----------------------------------------------------
    story.append(Paragraph("4. Transfer Learning (ResNet-18) & Ablation Study", h1_style))
    transfer_text = (
        "We apply transfer learning using a pretrained ResNet-18 model. The network is trained in two phases:\n"
        "Phase 1: Freeze all backbone layers, training only the classification head for 3 epochs (Adam, lr=1e-3).\n"
        "Phase 2: Unfreeze the last 2 convolutional blocks (layers 3 and 4), training for 5 more epochs with a 10x reduced learning rate (Adam, lr=1e-4).\n"
        "An ablation study evaluates the performance impact of unfreezing the convolutional blocks compared to keeping them frozen."
    )
    story.append(Paragraph(transfer_text, body_style))
    
    # Read Ablation text
    story.append(Spacer(1, 10))
    story.append(Paragraph("Frozen vs Unfrozen ResNet-18 Validation Results:", ParagraphStyle('SubH4', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, spaceAfter=8)))
    ablation_text = read_file_content(os.path.join(RESULTS_DIR, "transfer_ablation_study.txt"))
    for line in ablation_text.splitlines():
        story.append(Paragraph(line.replace(" ", "&nbsp;"), code_style))
        
    # Embed EuroSAT CM
    story.append(Spacer(1, 10))
    story.append(Paragraph("EuroSAT Validation Confusion Matrix (Unfrozen Model):", ParagraphStyle('SubH5', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, spaceAfter=6)))
    story.append(get_image(os.path.join(RESULTS_DIR, "transfer_eurosat_cm.png"), 320, 240))
    
    story.append(PageBreak())
    
    # ----------------------------------------------------
    # PAGE 5: UC MERCED HOLDOUT & ERROR ANALYSIS
    # ----------------------------------------------------
    story.append(Paragraph("5. UC Merced Holdout Evaluation & Error Analysis", h1_style))
    ucm_intro = (
        "To test domain generalization, the fine-tuned ResNet-18 model is evaluated on the UC Merced Land Use "
        "holdout dataset. The 21 classes are mapped semantically to EuroSAT's 10 classes. The performance drop "
        "quantifies the domain shift from Sentinel-2 satellite imagery (10m resolution) to USGS aerial photography (0.3m resolution)."
    )
    story.append(Paragraph(ucm_intro, body_style))
    
    # Embed UC Merced CM
    story.append(Spacer(1, 10))
    story.append(get_image(os.path.join(RESULTS_DIR, "transfer_uc_merced_cm.png"), 320, 240))
    
    # Error analysis and failure hypotheses
    story.append(Spacer(1, 15))
    story.append(Paragraph("Top-5 Error Analysis and Hypotheses:", ParagraphStyle('SubH6', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, spaceAfter=8)))
    error_analysis = (
        "1. <b>Golf Course classified as Pasture/Forest</b>: Golf courses feature extensive manicured grass and trees, "
        "making them spectrally identical to Pastures or Forests. Hypothesis: The model lacks contextual understanding of course layouts.<br/>"
        "2. <b>Freeway/Runway classified as Highway</b>: Roads, runways, and freeways are all asphalt structures. "
        "Hypothesis: Spectral similarity of asphalt dominates, ignoring geometric cues (e.g. runways have different line markups).<br/>"
        "3. <b>Baseball Diamond classified as Residential/Industrial</b>: Baseball fields are surrounded by urban spaces or features. "
        "Hypothesis: Spatial context from the surrounding area bleeds into features, causing urban misclassification.<br/>"
        "4. <b>Harbor classified as River/Industrial</b>: Harbors contain boats and water, surrounded by docks. "
        "Hypothesis: Water absorption spectra dominate, classifying the water as River and the structures as Industrial.<br/>"
        "5. <b>Chaparral classified as Herbaceous Vegetation</b>: Chaparral consists of shrubby bushes. "
        "Hypothesis: The high resolution of UCM highlights individual shrubs, which are averaged out in 10m EuroSAT pixels as generic grass/vegetation."
    )
    story.append(Paragraph(error_analysis, body_style))
    
    story.append(PageBreak())
    
    # ----------------------------------------------------
    # PAGE 6: EMBEDDING-BASED CHANGE DETECTION
    # ----------------------------------------------------
    story.append(Paragraph("6. Embedding-based Cosine Similarity Change Detector", h1_style))
    change_intro = (
        "The fine-tuned ResNet-18 backbone acts as a feature extractor (stripping the final FC classification layer) "
        "producing 512-dimensional embeddings. We simulate a time series using 50 geographic region grids (6x6 tiles) "
        "where 80% cells represent 'No Change' (different images of the same class) and 20% represent 'Change' (images of different classes). "
        "Change detection is formulated by thresholding the Cosine Similarity between corresponding tiles. We generate "
        "a ROC curve to evaluate performance."
    )
    story.append(Paragraph(change_intro, body_style))
    
    # Embed ROC
    story.append(Spacer(1, 10))
    story.append(get_image(os.path.join(RESULTS_DIR, "change_roc_curve.png"), 320, 240))
    
    # Print threshold points
    story.append(Spacer(1, 10))
    story.append(Paragraph("Operating Threshold Points:", ParagraphStyle('SubH7', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, spaceAfter=6)))
    thresh_text = read_file_content(os.path.join(RESULTS_DIR, "change_detection_thresholds.txt"))
    for line in thresh_text.splitlines():
        story.append(Paragraph(line.replace(" ", "&nbsp;"), code_style))
        
    story.append(PageBreak())
    
    # ----------------------------------------------------
    # PAGE 7: SPATIAL LEAKAGE & IMBALANCE EXPERIMENTS
    # ----------------------------------------------------
    story.append(Paragraph("7. Experiments: Spatial Leakage & Class Imbalance", h1_style))
    
    story.append(Paragraph("Spatial Leakage Gap Analysis:", ParagraphStyle('SubH8', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, spaceAfter=6)))
    leakage_text = read_file_content(os.path.join(RESULTS_DIR, "spatial_leakage_report.txt"))
    for line in leakage_text.splitlines():
        story.append(Paragraph(line, body_style))
        
    story.append(Spacer(1, 15))
    story.append(Paragraph("Class Imbalance Downsampling Analysis:", ParagraphStyle('SubH9', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, spaceAfter=6)))
    imbalance_text = read_file_content(os.path.join(RESULTS_DIR, "class_imbalance_report.txt"))
    for line in imbalance_text.splitlines():
        story.append(Paragraph(line, body_style))
        
    story.append(PageBreak())
    
    # ----------------------------------------------------
    # PAGE 8: EMBEDDING CLUSTERING & DASHBOARD
    # ----------------------------------------------------
    story.append(Paragraph("8. Embedding Representation & Interactive Geo-Dashboard", h1_style))
    story.append(Paragraph(
        "To evaluate representation learning quality, we project all validation set embeddings to 2D using t-SNE "
        "and compare the baseline scratch CNN features against our fine-tuned ResNet-18. ResNet-18 demonstrates "
        "substantially tighter, well-separated cluster boundaries, highlighting the effectiveness of pretrained transfer learning.",
        body_style
    ))
    
    # Embed t-SNE plot
    story.append(Spacer(1, 10))
    story.append(get_image(os.path.join(RESULTS_DIR, "embedding_tsne_comparison.png"), 480, 240))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("Sample Region Change Detection Heatmap (from change detector):", ParagraphStyle('SubH10', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, spaceAfter=6)))
    story.append(get_image(os.path.join(RESULTS_DIR, "region_1_change_heatmap.png"), 480, 200))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Interactive Geo-Dashboard:</b> The final system is packaged into a local Streamlit dashboard. "
        "Users can upload T1/T2 images, view Predicted Land Use classes and confidence scores, inspect embedding "
        "Cosine Similarity, toggling the operating threshold to see dynamic change maps. It also includes "
        "GradCAM visual overlays showing which features drove the classification.",
        body_style
    ))
    
    doc.build(story)
    print(f"Successfully generated PDF report at {PDF_PATH}")

if __name__ == "__main__":
    main()
