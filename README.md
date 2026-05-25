## Object-Based Visual SLAM for Urban Tram Navigation

This repository contains the code, analysis, and supporting materials for the project:

**Towards Object-Based Visual SLAM: A Revolution for Urban Tram Navigation**

The goal of this project is to design and validate an **Object-Based Visual SLAM pipeline** tailored to dynamic urban tram environments. We investigate how Big Data and deep-learning-based perception can be tightly fused with classical SLAM geometry to improve localisation accuracy, mapping cleanliness, and robustness against dynamic obstacles (vehicles, pedestrians, cyclists) and infrastructure (traffic lights, stations) in cluttered city corridors.

The end-to-end pipeline follows four stages:

1. **Stage 1 — Object Detection (YOLOv8).** Train and cross-evaluate YOLOv8n / YOLOv8x on **BDD100K** and **KITTI Object Detection** to obtain a robust detector that generalises across urban driving domains.
2. **Stage 2 — Multi-Object Tracking & 3D Cuboid Generation.** Extract 2D bounding boxes on KITTI odometry sequence 08, associate them across frames with a simplified **ByteTrack** policy, and lift them to oriented **3D cuboids** using a CubeSLAM-style monocular reasoning module.
3. **Stage 3 — ORB-SLAM3 Integration.** Inject the 3D cuboids into a modified **ORB-SLAM3** (monocular) running inside a Docker container, with live Pangolin visualisation forwarded to the host via XQuartz.
4. **Stage 4 — Evaluation.** Quantitative trajectory metrics (**ATE / RPE** via `evo`) and 3D detection metrics (**3D IoU**, precision/recall/F1) on KITTI Sequence 08, plus a qualitative real-world deployment on **Adelaide tram-route footage** captured on an iPhone.

This repository contains the code, configs, notebooks, Docker setup, and supporting materials for all four stages — together with the trained YOLOv8 weights, processed datasets, and evaluation artefacts that back the accompanying thesis.

> **Platform note.** This work was developed on **macOS (Apple Silicon, M1 Pro, 16 GB RAM)**. Some configuration steps below (Homebrew paths, XQuartz, `linux/arm64` Docker platform, `evo` `macosx` backend) are macOS-specific. Linux and Windows users will need to adapt the OpenCV/Pangolin paths, X-server, and Docker `--platform` flags accordingly.

---

## Project Components

The repository is organised around the four pipeline stages. The most relevant entry points are summarised below.

### Notebooks (`notebooks/`)

- `SLAM.ipynb` — **Stage 1 EDA & preprocessing.** Data loading, visualisation (bar charts, heatmaps, co-occurrence matrices), bounding-box statistics, annotation overlays, data cleaning (faulty boxes, outliers, occlusion/truncation), and conversion of BDD100K / KITTI to YOLO format.

  > **Note**: The `SLAM.ipynb` file is too large to render in the GitHub UI. Download the raw file to inspect and run it locally.

- `VSLAM_Colab_Training.ipynb` — **Stage 1 training & cross-dataset evaluation.** Augmentation, hyperparameters, YOLOv8n / YOLOv8x training on BDD100K, and in-domain (BDD→BDD) / cross-domain (BDD→KITTI) evaluation.

- `SLAM_Integration.ipynb` — **Stages 2–4 driver notebook.** Drives 2D bbox extraction with the trained YOLOv8x weights, applies the simplified ByteTrack policy, generates 3D cuboids, prepares the Adelaide iPhone sequence, and orchestrates the evaluation scripts under `utils/slam_integration/`.

### Stage-2/3/4 utilities (`utils/`)

- `utils/object_detector/` — `data_loader.py`, `helper_funcs.py` used by the EDA / training notebooks.
- `utils/slam_integration/`:
  - `cuboid_utils.py` — CubeSLAM-style 2D→3D cuboid reasoning (pinhole back-projection, class-specific priors, yaw recovery).
  - `preprocess_iphone_video.py` — Converts raw Adelaide iPhone clips into a KITTI-style sequence (RGB / grayscale frames, `times.txt`, intrinsics, `iPhone13.yaml`).
  - `fix_pose_mismatch.py` — Aligns ORB-SLAM3 `KeyFrameTrajectory.txt` with KITTI ground-truth poses for fair evaluation.
  - `analyze_ate_results.py`, `analyze_rpe_results.py` — Post-process `evo` zips into the publication-ready ATE/RPE plots.
  - `evaluate_3d_detection.py`, `run_evaluation_pipeline.py`, `diagnose_evaluation_issue.py` — End-to-end 3D IoU evaluation between predicted cuboids and ground-truth cuboids.
  - `calculate_reduction.py` — Quantifies cuboid de-duplication once ByteTrack is enabled.
  - `visualize_cuboids.py`, `remove_trailing_whitespace.py` — Helpers.

### ORB-SLAM3 integration (`third_party/`)

- `third_party/ORB_SLAM3/` — Cloned and **modified** UZ-SLAMLab/ORB_SLAM3 with our cuboid integration patches to `MapDrawer`, `FrameDrawer`, `System`, `Viewer`, and `Examples/Monocular/mono_kitti.cc`.
- `third_party/Pangolin/` — Pinned to **v0.6** for ARM64/Docker stability.

### Containerisation

- `Dockerfile` — `arm64v8/ubuntu:20.04` base with OpenCV 4.4.0, Pangolin v0.6, Eigen, GLEW, Mesa, X11 forwarding.
- `docker-compose.yml` — Mounts `third_party/ORB_SLAM3` and `data/` as bind mounts so source-only edits don't trigger a base-image rebuild.

### Data, results, and assets

- `data/` — KITTI odometry gray sequence 08, KITTI poses, processed Adelaide sequences, plus per-frame `.json` outputs: `bbox_outputs*`, `cuboid_outputs*` (static + ByteTrack + Adelaide variants), and `predicted_cuboids/` exported by ORB-SLAM3.
- `results/` — ATE, RPE, IoU evaluation artefacts (`results/ate/`, `results/rpe/`, `results/iou/`), `results/3d_map/` exported point clouds, and `results/yolov8n` / `results/yolov8s` training outputs.
- `pictures/` — Diagrams used by this README (`3D_cuboids_integration.png`, `mounting_volumes.png`, `pangolin_issue.png`, `orbslam3_running/*.png`).

## Setup Instructions

1. **Clone this repository**
   ```bash
   git clone https://github.com/your-username/Object-based-Visual-SLAM.git
   cd Object-based-Visual-SLAM
   ```

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

## Implementation Workflow

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

### 🗂️ Notes on Directory Contents

- `yolov8n_output/` and `yolov8s_output/`
→ These are automatically generated during training. You do not need to prepare them in advance. You may safely ignore `yolov8*_output/` when first setting up the project.

- `BACKUP/`
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
- Extract the zip files into their respective subdirectories to match the structure above. The extraction code is already prepared in `VSLAM_Colab_Training.ipynb` - simply uncomment and run the relevant !unzip commands for the files you want to extract.
> 🔁 **Make sure each image has a corresponding `.txt` file with the same base filename.** This ensures YOLOv8 can correctly match images with annotations during training.

You must extract the zipped files inside the corresponding images/ and labels/ subdirectories as shown above.

In `VSLAM_Colab_Training.ipynb`, train YOLOv8 models using:

- 📄 `bdd100k.yaml` — BDD100K dataset (in-domain training and testing)
- 📄 `bdd100k_kitti.yaml` — for training on BDD100K and testing on KITTI (cross-domain evaluation)

The notebook includes:
- Hyperparameter definitions (epochs, batch size, image size)
- Data augmentation configs (flipping, HSV jitter, cropping)
- Model checkpoints and logs

**Training Setup**: Choose GPU T4 with enhanced RAM option if you have Colab Pro. If using the free version, you may need to comment out `device=0` in the training code since it specifically targets GPU usage. Once configured, you're ready to train the model - simply click "Run All" and monitor the results after training completes.

---

#### ➤ Step 4: Evaluate Performance
All training outputs and visualisations are saved in the `results/` directory, including:
- Confusion matrices
- Validation batch prediction images
- Quantitative scores (mAP50, mAP50–95)

**Note**: The `results/` directory in this GitHub repository contains only selected results needed for the accompanying report, not the complete training outputs. For full inspection of all generated files, refer to the `yolov8n_output/` and `yolov8s_output/` directories mentioned in the [🗂️ Notes on Directory Contents](#🗂️-notes-on-directory-contents) section.

The complete results after training will look like the structure shown in the image below.

![Result Structure](https://i.imgur.com/e462hea.png)

Example findings:
- `mAP50`: Evaluates if predicted box overlaps ≥ 50% with ground truth
- `mAP50–95`: Averages mAP over IoU thresholds from 50% to 95%
- Cross-dataset generalisation: E.g., YOLOv8 trained on **BDD100K** and evaluated on **KITTI** to assess robustness

In our cross-dataset protocol, **YOLOv8x trained on BDD100K and evaluated on KITTI (Exp 2)** is the weight we propagate downstream — it sacrifices ~5 mAP50 versus the in-domain run but is more reliable on unseen urban geometry, which is exactly the regime our SLAM pipeline operates in.

---

## Stage 2 — Multi-Object Tracking & 3D Cuboid Generation

Stage 2 turns per-frame detections into temporally-consistent 3D landmarks. It runs entirely from `notebooks/SLAM_Integration.ipynb` and helpers in `utils/slam_integration/`, and emits one JSON file per KITTI frame.

> **Analogy.** Stage 1 takes a still photograph of every car on the street. Stage 2 (a) gives each car a name tag that persists across frames (ByteTrack), then (b) *inflates* the flat 2D photo into a 3D Lego brick (cuboid) the SLAM map can store.

#### ➤ Step 2.1 — 2D Bounding-Box Extraction

Run inference on KITTI odometry sequence 08 (4 071 frames) with the YOLOv8x weights from Stage 1, Exp 2:
- Confidence threshold: `0.6`
- IoU NMS threshold: `0.45`
- Output: one `xxxxxx.json` per frame in `data/bbox_outputs/` with `(x_min, y_min, x_max, y_max, class, conf)`.

#### ➤ Step 2.2 — Simplified ByteTrack Multi-Object Tracking

Instead of full Kalman-filter ByteTrack, we use a non-parametric two-stage IoU matcher:

- **High-confidence pass** (`conf ≥ τ_h = 0.6`) — matched against existing tracks via the **Hungarian algorithm** on an IoU cost matrix.
- **Low-confidence pass** (`τ_l = 0.1 ≤ conf < τ_h`) — recovers partially occluded objects by re-matching unmatched tracks against leftover detections.

Output: per-frame JSONs in `data/bbox_outputs_bytetrack/` augmented with persistent `track_id`s — these IDs are what ORB-SLAM3 keys its cuboid registry on.

#### ➤ Step 2.3 — 3D Cuboid Generation (CubeSLAM-style)

`utils/slam_integration/cuboid_utils.py` lifts each 2D box to a 9-DoF cuboid `O = {[R|t], [dx, dy, dz]}`:
1. Estimate depth from box height + class-specific real-world size priors (car / pedestrian / cyclist / truck).
2. Recover yaw by fitting the projected cuboid width to the 2D box width.
3. Pinhole back-project the cuboid centre into the camera frame, then transform to world coordinates using ORB-SLAM3's current pose.

Output: `data/cuboid_outputs/` (static baseline) and `data/cuboid_outputs_bytetrack/` (with tracking IDs) — each JSON contains 6-DoF pose + dimensions per cuboid, ready to be consumed by the modified `mono_kitti`.

![Stage 2 → 3 cuboid integration diagram](https://imgur.com/a/DtacfYu.png)
<!-- pictures/3D_cuboids_integration.png -->

---

## Stage 3 — ORB-SLAM3 Integration

Stage 3 fuses the 3D cuboids into a modified ORB-SLAM3 (monocular) and renders them alongside ORB feature points and keyframes inside Pangolin.

### 3.1 — Native build on macOS M1 (first attempt)

We initially built ORB-SLAM3 natively on macOS to keep the dev loop fast.

```bash
# Clone our patched fork (already vendored under third_party/)
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git third_party/ORB_SLAM3
cd third_party/ORB_SLAM3

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DOpenCV_DIR=/opt/homebrew/Cellar/opencv/4.12.0_11/lib/cmake/opencv4 \
         -DCMAKE_PREFIX_PATH=/opt/homebrew
make -j$(sysctl -n hw.logicalcpu)
```

`CMakeLists.txt` was patched to pin OpenCV ≥ 4.4 and use Homebrew's prefix:

```cmake
cmake_minimum_required(VERSION 3.5)
set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv4")
find_package(OpenCV 4.4 QUIET)
if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV > 4.4 not found.")
endif()
```

Whenever we edit anything under `Examples/Monocular/`, we rebuild:

```bash
cd third_party/ORB_SLAM3/build
make clean   # optional
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DOpenCV_DIR=/opt/homebrew/Cellar/opencv/4.12.0_11/lib/cmake/opencv4
make -j$(sysctl -n hw.logicalcpu)
```

To run on KITTI sequence 08 natively:

```bash
rm -f KeyFrameTrajectory.txt MapPoints.txt
cd ../Examples/Monocular
./mono_kitti ../../Vocabulary/ORBvoc.txt KITTI08.yaml \
  ../../../../data/data_odometry_gray/sequences/08
```

> **Known monocular caveat.** `SLAM.SaveTrajectoryKITTI()` errors out with *"cannot be used for monocular"*. ORB-SLAM3 saves monocular trajectories in TUM format only, so we convert TUM → KITTI manually before feeding `evo`.

### 3.2 — Pangolin failure on Apple Silicon → Docker

The native build crashed inside Pangolin with `NSInternalInconsistencyException` because macOS requires all UI calls on the main thread, while Pangolin's viewer thread is not the main thread under AppKit on Apple Silicon. To stabilise the environment, we containerised the entire toolchain.

![Pangolin NSInternalInconsistencyException on Apple Silicon](https://i.imgur.com/PLACEHOLDER_pangolin_issue.png)
<!-- pictures/pangolin_issue.png -->

**Prerequisites**
- Docker Desktop (with `linux/arm64` emulation enabled)
- [XQuartz](https://www.xquartz.org/) with **"Allow connections from network clients"** turned on
- `defaults write org.xquartz.X11 enable_iglx -bool true` (then restart XQuartz)
- In XQuartz xterm: `xhost + 127.0.0.1`

**Build the base image (one-time, ≈ 20 min)**

```bash
docker build --platform linux/arm64 -t orbslam3-base:latest .
```

The Dockerfile pins **OpenCV 4.4.0**, **Pangolin v0.6**, and omits non-essential packages (e.g. `libjasper-dev`) that break the ARM64 build.

**One-off run (no source mount)**

```bash
docker run -it --rm --platform linux/arm64 \
  -e DISPLAY=host.docker.internal:0 --network host \
  --volume "$(pwd)/data:/data" orbslam3-viz bash
```

### 3.3 — Mounting ORB-SLAM3 as a volume (the everyday workflow)

Once the base image exists, we keep the source tree on the host and bind-mount it into the container — so editing `mono_kitti.cc` or `MapDrawer.cc` only requires re-running `make`, never `docker build`.

```bash
# Start (or restart) the dev container
docker-compose up -d
docker-compose ps

# Enter the container
docker-compose exec orbslam3 bash

# Inside the container, rebuild as needed
cd /workspace/ORB_SLAM3/build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
```

![Docker bind-mounts (X11 socket, ORB-SLAM3 source, data)](https://i.imgur.com/PLACEHOLDER_mounting_volumes.png)
<!-- pictures/mounting_volumes.png -->

**Restarting cleanly:**

```bash
docker-compose down            # stop + remove old container
docker-compose up -d            # recreate with current compose config
docker-compose exec orbslam3 bash
```

### 3.4 — Running ORB-SLAM3 with 3D-cuboid integration

From inside the container (`cd /workspace/ORB_SLAM3/Examples/Monocular/`):

```bash
# Experiment 1 — Static 3D cuboid landmarks (no track IDs)
LIBGL_ALWAYS_INDIRECT=1 ./mono_kitti ../../Vocabulary/ORBvoc.txt KITTI08.yaml \
  /data/data_odometry_gray/sequences/08 /data/cuboid_outputs

# Experiment 2 — Dynamic Multi-Object Tracking (ByteTrack IDs)
LIBGL_ALWAYS_INDIRECT=1 ./mono_kitti ../../Vocabulary/ORBvoc.txt KITTI08.yaml \
  /data/data_odometry_gray/sequences/08 /data/cuboid_outputs_bytetrack
```

The fourth argument is the **cuboid directory** consumed by our patched `mono_kitti.cc`. Per-frame predicted cuboids are written to `/data/predicted_cuboids/` for Stage 4 IoU evaluation.

#### What we changed in ORB-SLAM3

Data flow of the integration: `mono_kitti → System → Viewer → MapDrawer / FrameDrawer`.

- **MapDrawer** — new `Cuboid` storage (`map<frameId, vector<Cuboid>>`), `SetCuboidsForFrame()`, `ClearOldCuboids(n)` sliding window of 30 frames, `DrawCuboids()` rendering wireframes in OpenGL with class-based RGB (red = vehicles, green = cyclists, blue = pedestrians) and camera-to-world transform.
- **FrameDrawer** — thread-safe `SetCuboids()` / `GetCuboids()` using `unique_lock` + `mutex mMutexCuboids`.
- **System** — pre-tracking call distributes cuboids from `mAllCuboids[idx]` to drawers; post-tracking increments `mCurrentFrameIdx` to keep frame indices aligned with detections.
- **Viewer** — rendering loop now invokes `DrawCuboids()` per frame.
- **mono_kitti.cc** — parses ByteTrack `track_id`s for persistent IDs and serialises predicted cuboids to JSON.

Qualitative outputs of the three experiments live in `pictures/orbslam3_running/`:

![Experiment 1 — Static cuboid landmarks](https://i.imgur.com/PLACEHOLDER_origin_cub.png)
<!-- pictures/orbslam3_running/origin_cub.png -->

![Experiment 2 — Dynamic ByteTrack tracking](https://i.imgur.com/PLACEHOLDER_bytetrack_cub.png)
<!-- pictures/orbslam3_running/bytetrack_cub.png -->

![Experiment 3 — Adelaide tram route](https://i.imgur.com/PLACEHOLDER_adelaide_cub_1.png)
<!-- pictures/orbslam3_running/adelaide_cub_1.png -->

#### Troubleshooting (Docker + XQuartz)

| Symptom | Fix |
|---|---|
| `xcb`/`GLX` errors at startup | Enable indirect GLX (`defaults write org.xquartz.X11 enable_iglx -bool true`), prefix runs with `LIBGL_ALWAYS_INDIRECT=1`, verify with `glxinfo` inside the container. |
| Pangolin window never opens | `xhost + 127.0.0.1` in XQuartz xterm; confirm `/tmp/.X11-unix` is bind-mounted. |
| `linux/amd64` image picked | Always pass `--platform linux/arm64`. |
| Pinned-version drift | Don't bump Pangolin past v0.6 or OpenCV past 4.4 — newer combos break on ARM64. |
| Disk space tight | `docker images | grep orbslam3`, then `docker rmi <unused_image_id>`. |

---

## Stage 4 — SLAM Evaluation

Stage 4 ships three experiments: **(1) Static cuboid baseline**, **(2) Dynamic ByteTrack tracking**, and **(3) Real-world Adelaide tram footage** (qualitative).

> **Activate the Python venv first** for any `evo_*` or `analyze_*` command:
>
> ```bash
> source venv/bin/activate
> ```
>
> One-time `evo` setup on macOS:
>
> ```bash
> evo_config set plot_backend macosx
> evo_config show
> ```

### 4.1 — Trajectory metrics (ATE & RPE)

We compare ORB-SLAM3's `KeyFrameTrajectory.txt` (TUM → KITTI converted) against KITTI ground-truth poses for sequence 08.

**Absolute Pose Error (ATE)** — global trajectory consistency:

```bash
# 1790 poses (main report figure)
evo_ape kitti ./data/data_odometry_poses/poses/08.txt \
  ./third_party/ORB_SLAM3/Examples/Monocular/KeyFrameTrajectory.txt \
  --correct_scale -a --pose_relation trans_part -va \
  --plot --plot_mode xz \
  --save_results results/ate_results.zip
```

`08_cleaned.txt` and `08_cleaned_downsampled.txt` (243 poses, `--plot_mode xyz`) variants are also supported when keyframe density differs.

**Relative Pose Error (RPE)** — local drift at 100 m segments:

```bash
evo_rpe kitti ./data/data_odometry_poses/poses/08_cleaned.txt \
  ./third_party/ORB_SLAM3/Examples/Monocular/KeyFrameTrajectory.txt \
  --delta 100 --delta_unit m --pose_relation trans_part \
  --align --correct_scale -va \
  --plot --plot_mode xyz \
  --save_results results/rpe_100m.zip
```

If the number of poses in `KeyFrameTrajectory.txt` doesn't match the GT file (typical for monocular):

```bash
python3 utils/slam_integration/fix_pose_mismatch.py \
  ./data/data_odometry_poses/poses/08.txt \
  ./third_party/ORB_SLAM3/Examples/Monocular/KeyFrameTrajectory.txt \
  ./experiment_outputs
```

**Publication-ready ATE/RPE plots** are produced by:

```bash
# ATE — static baseline
python3 utils/slam_integration/analyze_ate_results.py results/ate/ate_results \
  'ORB-SLAM3 with Static 3D Cuboid Landmarks'

# ATE — ByteTrack
python3 utils/slam_integration/analyze_ate_results.py results/ate/ate_results \
  'ORB-SLAM3 with Dynamic Multi-Object Tracking'

# RPE — static baseline
python3 utils/slam_integration/analyze_rpe_results.py results/rpe/rpe_100m \
  'ORB-SLAM3 with Static 3D Cuboid Landmarks'

# RPE — ByteTrack
python3 utils/slam_integration/analyze_rpe_results.py results/rpe/rpe_100m \
  'ORB-SLAM3 with Dynamic Multi-Object Tracking'
```

**Headline results on KITTI Sequence 08:**

| Metric | Exp 1 (Static) | Exp 2 (ByteTrack) |
|---|---|---|
| APE Mean (m) | 61.20 | 63.71 |
| APE RMSE (m) | 76.15 | 78.02 |
| RPE Mean (m) | 51.08 | 52.25 |
| RPE RMSE (m) | 65.42 | 67.31 |

Trajectory accuracy is statistically indistinguishable — as expected, because ByteTrack's contribution is *map cleanliness*, not pose estimation. The 61–64 m APE reflects the well-known monocular scale ambiguity inherent to KITTI sequence 08.

### 4.2 — 3D Object Detection (3D IoU)

We compare predicted cuboids against the ground-truth cuboid set (generated from KITTI labels) per frame.

```bash
# Optional sanity check first
python3 utils/slam_integration/diagnose_evaluation_issue.py \
  --gt_dir ./data/cuboid_outputs \
  --pred_dir ./data/predicted_cuboids

# Experiment 1 — Static cuboid landmarks
python3 utils/slam_integration/run_evaluation_pipeline.py \
  --gt_dir ./data/cuboid_outputs --pred_dir ./data/predicted_cuboids \
  --output_dir ./results --sequence 08 \
  --title 'ORB-SLAM3 with Static 3D Cuboid Landmarks' --generate_report

# Experiment 2 — ByteTrack integration
python3 utils/slam_integration/run_evaluation_pipeline.py \
  --gt_dir ./data/cuboid_outputs_bytetrack --pred_dir ./data/predicted_cuboids \
  --output_dir ./results --sequence 08 \
  --title "ORB-SLAM3 with Dynamic Multi-Object Tracking" --generate_report

# Experiment 3 — Adelaide
python3 utils/slam_integration/run_evaluation_pipeline.py \
  --gt_dir ./data/cuboid_outputs_adelaide --pred_dir ./data/predicted_cuboids \
  --output_dir ./results --sequence adelaide \
  --title "ORB-SLAM3 on Real-World Adelaide Dataset with Dynamic Multi-Object Tracking" \
  --generate_report
```

**Headline 3D IoU on KITTI Sequence 08:**

| Metric | Exp 1 (Static) | Exp 2 (ByteTrack) |
|---|---|---|
| Mean 3D IoU | 0.519 | 0.517 |
| Precision (IoU ≥ 0.25) | 0.650 | 0.650 |
| Recall (IoU ≥ 0.25) | 0.605 | 0.602 |
| F1 (IoU ≥ 0.25) | 0.627 | 0.625 |

These results match the SOTA range for monocular static-cuboid approaches on KITTI odometry. The IoU 0.25 threshold is the appropriate operating point for monocular SLAM because scale error follows a Gaussian distribution in log-space.

**Cuboid reduction** (the *real* benefit of ByteTrack):

```bash
python3 utils/slam_integration/calculate_reduction.py \
  ./data/cuboid_outputs_bytetrack ./data/predicted_cuboids
```

ByteTrack integration consolidates **12 214 frame-level detections into 2 298 unique tracked objects — an 81.2 % reduction in redundant map entries**, dramatically improving point-cloud cleanliness without sacrificing trajectory accuracy.

### 4.3 — Qualitative deployment on Adelaide tram footage

We captured raw iPhone 13 video on Adelaide tram corridors (`data/adelaide.mp4`, `data/adelaide_2.mp4`) and turned it into a KITTI-style sequence:

```bash
python3 utils/slam_integration/preprocess_iphone_video.py ./data/adelaide.mp4 \
  --output ./data/adelaide_data --fps 10 --gray
```

Resulting layout (under `data/adelaide_data/`):

```
processed_video/
├── rgb/             # Colour frames
├── gray/            # Grayscale frames (used by mono_kitti)
├── calibration/     # Intrinsic K matrices
├── times.txt        # Timestamps
└── iPhone13.yaml    # ORB-SLAM3 config tuned for iPhone 13
```

Then, inside the Docker container:

```bash
LIBGL_ALWAYS_INDIRECT=1 ./mono_kitti ../../Vocabulary/ORBvoc.txt Adelaide.yaml \
  /data/adelaide_sequence /data/cuboid_outputs_adelaide

LIBGL_ALWAYS_INDIRECT=1 ./mono_kitti ../../Vocabulary/ORBvoc.txt Adelaide.yaml \
  /data/adelaide_sequence_2 /data/cuboid_outputs_adelaide_2
```

The system successfully tracks trams, cars, and cyclists with persistent IDs across frames; minor false positives (e.g. Adelaide Metro bus misclassified as a truck, traffic-sign pole as a cyclist) are still excluded from the static map and therefore do not corrupt the point cloud.

![Adelaide qualitative result — busy tram corridor](https://i.imgur.com/PLACEHOLDER_adelaide_cub_3.png)
<!-- pictures/orbslam3_running/adelaide_cub_3.png -->

---

## Reproducing the Full Pipeline End-to-End

A minimal "from clean checkout to evaluation" walkthrough:

```bash
# 1. Clone and set up Python venv (Stage 1 + Stage 4 tooling)
git clone https://github.com/your-username/Object-based-Visual-SLAM.git
cd Object-based-Visual-SLAM
python3 -m venv venv && source venv/bin/activate
pip install pandas numpy matplotlib opencv-python tabulate evo ultralytics

# 2. Stage 1 — run SLAM.ipynb + VSLAM_Colab_Training.ipynb (Colab GPU recommended)

# 3. Stage 2 — run SLAM_Integration.ipynb to produce data/bbox_outputs* and data/cuboid_outputs*

# 4. Stage 3 — build Docker base, then bring the dev container up
docker build --platform linux/arm64 -t orbslam3-base:latest .
docker-compose up -d
docker-compose exec orbslam3 bash
# inside: cd /workspace/ORB_SLAM3/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
# inside: cd ../Examples/Monocular && LIBGL_ALWAYS_INDIRECT=1 ./mono_kitti ... /data/cuboid_outputs_bytetrack

# 5. Stage 4 — evaluate (on host, venv active)
evo_config set plot_backend macosx
evo_ape kitti ... && evo_rpe kitti ...
python3 utils/slam_integration/run_evaluation_pipeline.py ... --generate_report
```

---

## Limitations & Future Work

- **Monocular scale ambiguity** dominates the residual 61–64 m APE on KITTI 08 — stereo or visual-inertial extensions are the natural next step.
- **Detector ceiling** — YOLOv8x cross-domain mAP50 of ~40 % caps cuboid recall; upgrading to YOLO26 (released Sep 2025) or fine-tuning on Adelaide-collected GT would lift Stage 2 quality.
- **Cuboid optimisation** — we ship a simplified CubeSLAM-style reasoning module; integrating the full multi-view bundle adjustment from `Algorithm 2` (Yang & Scherer, 2019) into ORB-SLAM3's back-end is left as future work.
- **Adelaide quantitative metrics** — qualitative only for now; a labelled Adelaide tram benchmark would let us close the loop quantitatively.