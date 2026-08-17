# Computer-Vision-Projects
This repository contains five computer vision and perception projects I built during my Computer Vision Project (Course at FAU). The core theme across these repositories is building robust optical pipelines from scratch—focusing on raw physical data processing, temporal tracking, and spatial mathematics rather than just plugging into black-box APIs. 

Below is a detailed breakdown of the engineering challenges and technical implementations for each project.

---

## 1. 3D Scene Perception & Bounding Box Estimation
**The Challenge:** Time-of-Flight (ToF) depth sensors output incredibly noisy 3D point clouds. The goal was to filter this noise, remove the background, and accurately estimate the 3D dimensions (bounding boxes) of physical objects in the scene.

**The Pipeline:**
*   **Noise Filtering:** Applied morphological filters and isolated the primary objects using Connected Component Analysis.
*   **Ground Plane Removal:** Leveraged MLESAC (Maximum Likelihood Estimation SAmple Consensus)—a robust variant of RANSAC—to find and subtract the ground plane from the 3D point cloud.
*   **Real-Time Constraints:** Standard RANSAC was too computationally heavy, so I implemented a custom *Preemptive RANSAC* strategy. This evaluated hypotheses on a small subset of data first, instantly discarding poor fits and drastically reducing compute time.

**The Result:** Successfully mapped 2D coordinate projections into accurate 3D Euclidean space, extracting precise object dimensions with minimal computational overhead.

---

## 2. Computational Imaging: RAW Sensor Processing & Demosaicing
**The Challenge:** Modern cameras hide their image signal processing (ISP) behind proprietary APIs. This project required building a complete computational photography pipeline from scratch to reconstruct a viewable RGB image directly from raw multidimensional `.npy` sensor arrays.

**The Pipeline:**
*   **CRF Modeling:** Mathematically modeled the Camera Response Function (CRF) using non-linear least squares optimization. This allowed the pipeline to recover true, physical light values and linearize the image space.
*   **Bayer Demosaicing:** Built custom demosaicing algorithms to interpolate the raw sensor grid into full RGB channels.
*   **Tone Mapping:** Applied non-linear tone mapping to compress the high dynamic range into a viewable format.

**The Result:** Successfully built a full ISP pipeline in Python. The biggest engineering hurdle was strictly managing floating-point precision across massive multidimensional arrays to prevent data corruption during the non-linear transformations.


---

## 3. Large-Scale Writer Retrieval
**The Challenge:** Identifying specific handwriting styles from the massive ICDAR17 dataset using classical computer vision descriptors and advanced pooling techniques.

**The Pipeline:**
*   **Feature Extraction:** Extracted highly discriminative Upright RootSIFT descriptors from raw handwriting patches.
*   **Classification:** Trained Exemplar-SVMs (One-vs-Rest) to classify individual writer styles.
*   **Ensemble & Pooling:** Represented the documents using PCA-whitened Multi-VLAD ensembles. To boost the discriminative power of the vectors, I implemented Generalized Max Pooling (GMP).

**The Result:** The advanced pooling and ensemble techniques boosted the baseline retrieval performance by 24%, achieving a highly competitive Mean Average Precision (mAP) of 0.78 on the dataset.


---

## 4. Open-Set Face Recognition & Tracking
**The Challenge:** Recognizing faces in a continuous video feed is difficult, but it becomes significantly harder when you have to account for "unknown" individuals who were never in your training dataset (open-set recognition).

**The Pipeline:**
*   **Tracking & Re-initialization:** Engineered a temporal, frame-by-frame tracking system. If a face was lost due to a sudden pose change or occlusion, the system utilized custom re-initialization logic to re-acquire the target in subsequent frames.
*   **Handling Unknowns:** Implemented Single and Multi Pseudo Label (SPL/MPL) strategies to dynamically assign labels to unknown identities, keeping them separated from the known users in the database.
*   **Optimization:** Applied dynamic K-Means clustering to the unknown identities and replaced naive distance calculations with vectorized Nearest Neighbor searches.

**The Result:** Achieved 90% validation accuracy on open-set identification while maintaining a pipeline fast enough for deployment.

---

## 5. Hierarchical Object Detection (Selective Search & R-CNN)
**The Challenge:** Building a classic R-CNN pipeline from the ground up to detect objects in cluttered environments, dealing specifically with severe data imbalance and false positive predictions.

**The Pipeline:**
*   **Region Proposals:** Generated bounding box hypotheses using multi-scale Selective Search. 
*   **Feature Extraction & Classification:** Passed the proposals through a pre-trained ResNet18 CNN to extract features, which were then fed into a custom RBF-Kernel Support Vector Machine (SVM).
*   **False Positive Filtering:** Addressed severe class imbalances by implementing balanced class weighting and hard negative mining. 
*   **Box Cleanup:** Applied rigorous Non-Maximum Suppression (NMS) and geometric data augmentation to clean up overlapping and duplicate bounding boxes.

**The Result:** The system achieved a Mean Average Best Overlap (MABO) of 0.74, proving high region proposal quality and strong classification robustness in cluttered scenes.


---

*Technical Stack across all projects: Python, PyTorch, OpenCV, Scikit-learn, NumPy, SciPy, Git, Docker.*
