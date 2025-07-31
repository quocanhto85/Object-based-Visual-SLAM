
## Object-Based Visual SLAM for Urban Tram Navigation

This repository contains the code, analysis, and supporting materials for the project:

**Towards Object-Based Visual SLAM: A Revolution for Urban Tram Navigation**

The goal is to investigate how Big Data and deep learning-based object detection (YOLO) can enhance localisation accuracy in dynamic urban environments. This project includes exploratory data analysis (EDA), data cleaning, preprocessing, and modelling using two principal datasets: **KITTI Object Detection** and **BDD100K**.

---

## Project Components

- `SLAM.ipynb` — Main Jupyter Notebook containing:
  - Data loading
  - Visualisation (bar charts, heatmaps, co-occurrence matrices)
  - Bounding box statistics
  - Example image annotation overlays
  - Data cleaning and bounding box correction
  - Outlier and noise removal
  - Occlusion and truncation handling
  - YOLO-format conversion
  - Data augmentation and training setup

- `VSLAM_Colab_Training.ipynb` — Extended notebook for:
  - Traning and Evaluating on BDD100K and KITTI

## Setup Instructions

1. **Clone this repository**
   ```bash
   git clone https://github.com/your-username/Object-based-Visual-SLAM.git
   cd Object-based-Visual-SLAM

2. **Download and setup datasets**

- [KITTI Object Detection](https://datasetninja.com/kitti-object-detection)
- [BDD100K](https://datasetninja.com/bdd100k)

Download and extract the datasets into the main project directory (here referred to as the `\root` directory), ensuring that the extracted files and subfolders match the layout presented in the figure below:

![Dataset Structure](https://i.imgur.com/oF1AoFB.png)

Subsequently, you can explore the content inside both datasets. Along with the main analysis code `SLAM.ipynb`, here is the overall structure of the project:

![Project Structure](https://i.imgur.com/HYSjSAD.png)


3. **Install dependencies**

Open the code with Visual Studio Code (or other IDEs that suit your needs) and install these libraries via command line:

   ```bash
   pip install pandas numpy matplotlib opencv-python tabulate
   ```

4. **Run the analysis and modelling pipeline**

 Finally, you are ready to explore the notebook. To re-run all cells, simply click the `Run All`. The analytical findings are summarised in the accompanying report.

![Code](https://i.imgur.com/Cc5bq7R.png)

#### ➤ Step 1: Exploratory Data Analysis (EDA)
Use `SLAM.ipynb` to explore and visualise both datasets. This includes:
- Distribution of object types
- Heatmaps of object locations
- Annotation consistency checks
- Spatial co-occurrence matrices

These insights help understand class imbalance and spatial priors before training.

---

#### ➤ Step 2: Data Cleaning and Preprocessing
Use `SLAM.ipynb` to perform:
- ✅ Correction of faulty bounding boxes (e.g., swapped `xmin > xmax`)
- ✅ Removal of small or noisy bounding boxes (e.g., area < 100 pixels²)
- ✅ Handling of occluded/truncated objects via classification filters
- ✅ Conversion of datasets into YOLO-compatible format (`.txt` with normalised center x/y, width, height)

This ensures clean and standardised annotations for robust model training.

---

#### ➤ Step 3: Model Training

Before running `VSLAM_Colab_Training.ipynb`, make sure your Google Drive directory is organised like this:

![Project Structure](https://i.imgur.com/krVHZZs.png)

> This structure ensures that all necessary outputs, config files, and checkpoints are properly saved and referenced during training and evaluation.

> Notes on Directory Contents

`yolov8n_output/` and `yolov8s_output/`
→ These are automatically generated during training. You do not need to prepare them in advance. You may safely ignore `yolov8*_output/` when first setting up the project.

`BACKUP/`
→ Used to store zipped versions of datasets, YAML files, or model weights for backup purposes.

### 📦 Dataset Preparation & Google Drive Structure

After running through the entire `SLAM.ipynb` notebook, you will obtain pre-processed annotations in YOLO format for both **BDD100K** and **KITTI** datasets in your local machine. 

Once generated, zip the folders locally and upload them to your Google Drive under the TRAINING directory.

````markdown
TRAINING/
├── images/
│   ├── train/
│   ├── val/
│   └── val_kitti/
└── labels/
    ├── train/
    ├── val/
    └── val_kitti/
````

📌 **Instructions**: 
- Zip the `images/` and `labels/` folders locally after generation. 
- Upload them to your Google Drive under the `TRAINING/` directory. 
- Extract the zip files into their respective subdirectories to match the structure above. > 🔁 **Make sure each image has a corresponding `.txt` file with the same base filename.** This ensures YOLOv8 can correctly match images with annotations during training.

You must extract the zipped files inside the corresponding images/ and labels/ subdirectories as shown above.

In `VSLAM_Colab_Training.ipynb`, train YOLOv8 models using:

- 📄 `bdd100k.yaml` — BDD100K dataset (in-domain training and testing)
- 📄 `bdd100k_kitti.yaml` — for training on BDD100K and testing on KITTI (cross-domain evaluation)

The notebook includes:
- Hyperparameter definitions (epochs, batch size, image size)
- Data augmentation configs (flipping, HSV jitter, cropping)
- Model checkpoints and logs

---

#### ➤ Step 4: Evaluate Performance
All training outputs and visualisations are saved in the `results/` directory, including:
- Confusion matrices
- Validation batch prediction images
- Quantitative scores (mAP50, mAP50–95)

Example findings:
- `mAP50`: Evaluates if predicted box overlaps ≥ 50% with ground truth
- `mAP50–95`: Averages mAP over IoU thresholds from 50% to 95%
- Cross-dataset generalisation: E.g., YOLOv8 trained on BDD100K and evaluated on KITTI to assess robustness