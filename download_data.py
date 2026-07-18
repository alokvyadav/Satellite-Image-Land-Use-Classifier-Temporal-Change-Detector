import os
import zipfile
import urllib.request
import ssl
import shutil

# Bypass SSL verification to avoid potential certificate errors
ssl._create_default_https_context = ssl._create_unverified_context

# Paths
WORKSPACE = "C:/Users/alok1/.gemini/antigravity-ide/scratch/satellite_change_detection"
DATA_DIR = os.path.join(WORKSPACE, "data")
SPLITS_DIR = os.path.join(WORKSPACE, "splits")

# Create dirs
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)

# URLs
EUROSAT_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
UCMERCED_URL = "https://huggingface.co/datasets/torchgeo/ucmerced/resolve/main/UCMerced_LandUse.zip"

SPLIT_URLS = {
    "eurosat-spatial-train.txt": "https://huggingface.co/datasets/torchgeo/eurosat/raw/main/eurosat-spatial-train.txt",
    "eurosat-spatial-val.txt": "https://huggingface.co/datasets/torchgeo/eurosat/raw/main/eurosat-spatial-val.txt",
    "eurosat-spatial-test.txt": "https://huggingface.co/datasets/torchgeo/eurosat/raw/main/eurosat-spatial-test.txt"
}

def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    def report_hook(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = read_so_far * 100 / total_size
            print(f"Downloaded {percent:.1f}% ({read_so_far / 1024**2:.1f} MB of {total_size / 1024**2:.1f} MB)", end="\r")
        else:
            print(f"Downloaded {read_so_far / 1024**2:.1f} MB", end="\r")
    urllib.request.urlretrieve(url, dest, reporthook=report_hook)
    print("\nDownload complete.")

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def main():
    # 1. Download splits
    for filename, url in SPLIT_URLS.items():
        dest = os.path.join(SPLITS_DIR, filename)
        if not os.path.exists(dest):
            download_file(url, dest)
        else:
            print(f"{filename} already exists.")

    # 2. Download EuroSAT
    eurosat_zip = os.path.join(DATA_DIR, "EuroSAT_RGB.zip")
    if not os.path.exists(eurosat_zip):
        download_file(EUROSAT_URL, eurosat_zip)
    else:
        print("EuroSAT_RGB.zip already exists.")

    eurosat_extract = os.path.join(DATA_DIR, "EuroSAT_RGB_temp")
    if not os.path.exists(os.path.join(DATA_DIR, "EuroSAT_RGB")):
        extract_zip(eurosat_zip, eurosat_extract)
        # EuroSAT unzips into 'EuroSAT_RGB/' subfolder
        temp_folder = os.path.join(eurosat_extract, "EuroSAT_RGB")
        if os.path.exists(temp_folder):
            os.rename(temp_folder, os.path.join(DATA_DIR, "EuroSAT_RGB"))
            shutil.rmtree(eurosat_extract)
            print("EuroSAT RGB dataset successfully set up in data/EuroSAT_RGB")
        else:
            # Try 2750 folder just in case
            alt_folder = os.path.join(eurosat_extract, "2750")
            if os.path.exists(alt_folder):
                os.rename(alt_folder, os.path.join(DATA_DIR, "EuroSAT_RGB"))
                shutil.rmtree(eurosat_extract)
                print("EuroSAT RGB dataset successfully set up in data/EuroSAT_RGB (using 2750)")
            else:
                print(f"Error: expected 'EuroSAT_RGB' or '2750' directory inside {eurosat_extract}")
    else:
        print("data/EuroSAT_RGB already exists.")

    # 3. Download UC Merced
    ucmerced_zip = os.path.join(DATA_DIR, "UCMerced_LandUse.zip")
    if not os.path.exists(ucmerced_zip):
        download_file(UCMERCED_URL, ucmerced_zip)
    else:
        print("UCMerced_LandUse.zip already exists.")

    if not os.path.exists(os.path.join(DATA_DIR, "UCMerced_LandUse")):
        extract_zip(ucmerced_zip, DATA_DIR)
        print("UC Merced dataset successfully set up in data/UCMerced_LandUse")
    else:
        print("data/UCMerced_LandUse already exists.")

    print("\nAll datasets and split files downloaded and extracted successfully!")

if __name__ == "__main__":
    main()
