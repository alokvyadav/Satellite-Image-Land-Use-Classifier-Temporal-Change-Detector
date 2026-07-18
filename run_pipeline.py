import os
from data_pipeline import plot_class_distribution, visualize_class_samples

# Configurations
WORKSPACE = "C:/Users/alok1/.gemini/antigravity-ide/scratch/satellite_change_detection"
DATA_EUROSAT = os.path.join(WORKSPACE, "data", "EuroSAT_RGB")
SPLITS_DIR = os.path.join(WORKSPACE, "splits")
RESULTS_DIR = os.path.join(WORKSPACE, "results")

def main():
    print("Executing Data Pipeline analysis...")
    
    # 1. Plot Class Distribution
    dist_plot_path = os.path.join(RESULTS_DIR, "class_distribution.png")
    plot_class_distribution(DATA_EUROSAT, SPLITS_DIR, dist_plot_path)
    
    # 2. Visualize Class Samples
    visualize_class_samples(DATA_EUROSAT, RESULTS_DIR, n_samples=5)
    
    print("Data Pipeline analysis complete!")

if __name__ == "__main__":
    main()
