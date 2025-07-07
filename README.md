
## Object-Based Visual SLAM for Urban Tram Navigation

This repository contains the code, analysis, and supporting materials for the project:

**Towards Object-Based Visual SLAM: A Revolution for Urban Tram Navigation**

The goal is to investigate how Big Data and deep learning-based object detection (YOLO) can enhance localisation accuracy in dynamic urban environments. This project includes exploratory data analysis (EDA) and visualisation of two principal datasets: KITTI Object Dectection and BDD100K.

All analysis code is provided in:
- `SLAM.ipynb` — Main Jupyter Notebook containing:
  - Data loading
  - Visualisation (bar charts, heatmaps, co-occurrence matrices)
  - Bounding box statistics
  - Example image annotation overlays

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

4. **Run the analysis**

 Finally, you are ready to explore the notebook. To re-run all cells, simply click the `Run All`. The analytical findings are summarised in the accompanying report         (`Object_Based_VSLAM_Data_Analysis.pdf`) included in this repository.

![Code](https://i.imgur.com/Cc5bq7R.png)


