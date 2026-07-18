import os
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

from models import get_resnet18_model, ResNet18Embedder
from data_pipeline import get_transforms, EUROSAT_CLASSES

# Page Configuration
st.set_page_config(
    page_title="GeoEye: Satellite Land-Use Classifier & Change Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for premium dark/light mode integration
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1f4068, #162447, #e43f5a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #1f4068;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .change-flag-yes {
        background-color: #ffe6e6;
        color: #d63031;
        padding: 0.8rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        border: 1px solid #ffb3b3;
    }
    .change-flag-no {
        background-color: #e6ffe6;
        color: #27ae60;
        padding: 0.8rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        border: 1px solid #b3ffb3;
    }
</style>
""", unsafe_allow_html=True)

# Configurations
WORKSPACE = "C:/Users/alok1/.gemini/antigravity-ide/scratch/satellite_change_detection"
CHECKPOINT_DIR = os.path.join(WORKSPACE, "checkpoints")
RESULTS_DIR = os.path.join(WORKSPACE, "results")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_models():
    model = get_resnet18_model(num_classes=10, pretrained=False)
    ft_path = os.path.join(CHECKPOINT_DIR, "resnet18_fine_tuned.pt")
    
    if not os.path.exists(ft_path):
        ft_path = os.path.join(CHECKPOINT_DIR, "resnet18_frozen.pt")
        
    if os.path.exists(ft_path):
        model.load_state_dict(torch.load(ft_path, map_location=device))
        model.to(device)
        model.eval()
        embedder = ResNet18Embedder(model).to(device)
        embedder.eval()
        return model, embedder
    else:
        return None, None


# --- GradCAM Implementation ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        
        self.forward_hook = self.target_layer.register_forward_hook(self.save_features)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradients)
        
    def save_features(self, module, input, output):
        self.features = output
        
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        loss = output[0, class_idx]
        loss.backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.features, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
        
        # Normalize between 0 and 1
        heatmap = heatmap - torch.min(heatmap)
        heatmap = heatmap / (torch.max(heatmap) + 1e-8)
        
        return heatmap.squeeze().cpu().detach().numpy(), output
        
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def overlay_heatmap(original_image, heatmap, alpha=0.45):
    img_np = np.array(original_image.resize((224, 224)))
    h, w, c = img_np.shape
    
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    overlayed = cv2.addWeighted(img_np, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed


# --- Load Thresholds ---
def get_thresholds():
    # Default fallback values
    thresholds = {
        "Balanced": 0.85,
        "High Recall": 0.90,
        "High Precision": 0.80
    }
    
    file_path = os.path.join(RESULTS_DIR, "change_detection_thresholds.txt")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if "Similarity Threshold:" in line:
                    val = float(line.split()[-1])
                    if "Balanced" in lines[lines.index(line)-2]:
                        thresholds["Balanced"] = val
                    elif "High Recall" in lines[lines.index(line)-2]:
                        thresholds["High Recall"] = val
                    elif "High Precision" in lines[lines.index(line)-2]:
                        thresholds["High Precision"] = val
        except Exception as e:
            print(f"Error loading thresholds from file: {e}")
            
    return thresholds


def main():
    st.markdown('<div class="main-header">GeoEye: Satellite Land-Use & Change Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Classify satellite image land use and map temporal changes side-by-side.</div>', unsafe_allow_html=True)
    
    # Load model resources
    model, embedder = load_models()
    
    if model is None:
        st.warning("⚠️ Fine-tuned ResNet-18 checkpoint not found in `checkpoints/`. Please run the model training scripts first.")
        return
        
    # Get threshold configurations
    threshold_opts = get_thresholds()
    
    # Sidebar
    st.sidebar.title("🎛️ Configurations")
    operating_point = st.sidebar.selectbox(
        "Select Operating Point (Change Detection)",
        options=["Balanced", "High Recall", "High Precision"]
    )
    
    selected_threshold = threshold_opts[operating_point]
    st.sidebar.write(f"**Similarity Threshold:** `{selected_threshold:.3f}`")
    st.sidebar.markdown("""
    - **Balanced**: Balances false positives and false negatives (optimal F1).
    - **High Recall**: Flags change at a higher similarity limit to capture almost all true changes.
    - **High Precision**: Restricts flags only to highly different embeddings to minimize false alarms.
    """)
    
    # File uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Before Image (T1)")
        uploaded_file_t1 = st.file_uploader("Upload T1 Satellite Image...", type=["jpg", "jpeg", "png", "tif"])
        
    with col2:
        st.subheader("After Image (T2)")
        uploaded_file_t2 = st.file_uploader("Upload T2 Satellite Image...", type=["jpg", "jpeg", "png", "tif"])
        
    if uploaded_file_t1 and uploaded_file_t2:
        # Load images
        img_t1 = Image.open(uploaded_file_t1).convert('RGB')
        img_t2 = Image.open(uploaded_file_t2).convert('RGB')
        
        # Prepare transforms
        _, val_transform, _ = get_transforms()
        
        t1_tensor = val_transform(img_t1).unsqueeze(0).to(device)
        t2_tensor = val_transform(img_t2).unsqueeze(0).to(device)
        
        # --- Classification and Embeddings ---
        # T1 Pass
        model.eval()
        gradcam_t1 = GradCAM(model, model.layer4)
        
        # Get class indices
        with torch.no_grad():
            outputs_t1_raw = model(t1_tensor)
            probs_t1 = F.softmax(outputs_t1_raw, dim=1)
            conf_t1, idx_t1 = torch.max(probs_t1, 1)
            conf_t1, idx_t1 = conf_t1.item(), idx_t1.item()
            
        gradcam_t1.remove_hooks()
        
        # Recalculate with active hook for GradCAM
        gradcam_t1 = GradCAM(model, model.layer4)
        heatmap_t1, _ = gradcam_t1.generate_heatmap(t1_tensor, idx_t1)
        gradcam_t1.remove_hooks()
        
        # T2 Pass
        gradcam_t2 = GradCAM(model, model.layer4)
        with torch.no_grad():
            outputs_t2_raw = model(t2_tensor)
            probs_t2 = F.softmax(outputs_t2_raw, dim=1)
            conf_t2, idx_t2 = torch.max(probs_t2, 1)
            conf_t2, idx_t2 = conf_t2.item(), idx_t2.item()
            
        gradcam_t2.remove_hooks()
        
        gradcam_t2 = GradCAM(model, model.layer4)
        heatmap_t2, _ = gradcam_t2.generate_heatmap(t2_tensor, idx_t2)
        gradcam_t2.remove_hooks()
        
        # --- Extract Embeddings & Compute Cosine Similarity ---
        with torch.no_grad():
            emb_t1 = embedder(t1_tensor).cpu().numpy()[0]
            emb_t2 = embedder(t2_tensor).cpu().numpy()[0]
            
        # Cosine Similarity
        similarity = np.dot(emb_t1, emb_t2) / (np.linalg.norm(emb_t1) * np.linalg.norm(emb_t2) + 1e-8)
        
        # Display Side-by-side Predictions
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>T1 Prediction</h4>
                <h3>{EUROSAT_CLASSES[idx_t1]}</h3>
                <p>Confidence: <b>{conf_t1*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>T2 Prediction</h4>
                <h3>{EUROSAT_CLASSES[idx_t2]}</h3>
                <p>Confidence: <b>{conf_t2*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            
        # Display Cosine Similarity Metric & Change Flag
        is_change = similarity < selected_threshold
        
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("Embedding Cosine Similarity", f"{similarity:.4f}")
            
        with col_metric2:
            st.write("#### Change Status")
            if is_change:
                st.markdown('<div class="change-flag-yes">⚠️ LAND-USE CHANGE DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="change-flag-no">✅ NO SIGNIFICANT CHANGE</div>', unsafe_allow_html=True)
                
        # Show Images and GradCAM Overlays
        st.write("---")
        st.subheader("Visualization Analysis")
        
        tab1, tab2 = st.tabs(["Original Images", "GradCAM Feature Explanations"])
        
        with tab1:
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(img_t1, caption="Before Image (T1)", use_container_width=True)
            with col_img2:
                st.image(img_t2, caption="After Image (T2)", use_container_width=True)
                
        with tab2:
            col_cam1, col_cam2 = st.columns(2)
            
            # Draw GradCAM Overlays
            overlay_t1 = overlay_heatmap(img_t1, heatmap_t1)
            overlay_t2 = overlay_heatmap(img_t2, heatmap_t2)
            
            with col_cam1:
                st.image(overlay_t1, caption=f"GradCAM Overlay (Class: {EUROSAT_CLASSES[idx_t1]})", use_container_width=True)
            with col_cam2:
                st.image(overlay_t2, caption=f"GradCAM Overlay (Class: {EUROSAT_CLASSES[idx_t2]})", use_container_width=True)
                
    else:
        # Default screen: instruct the user
        st.info("💡 Upload both Before (T1) and After (T2) satellite images in the columns above to run classification, extract embeddings, compute similarities, and evaluate change.")

if __name__ == "__main__":
    main()
