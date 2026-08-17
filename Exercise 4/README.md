# Face Recognition and Detection System

A comprehensive face recognition and detection system built with OpenCV, MTCNN, and deep learning embeddings. This project supports both supervised face identification and unsupervised face clustering.

## Features

- **Real-time Face Detection**: Using MTCNN for robust face detection
- **Face Tracking**: Template matching-based face tracking across video frames
- **Face Recognition**: Supervised learning with k-NN classifier for identity recognition
- **Face Clustering**: Unsupervised k-means clustering for face grouping
- **Open-set Identification**: Handles unknown faces with confidence thresholding
- **Performance Evaluation**: DIR (Detection and Identification Rate) curve analysis

## Requirements

### System Requirements
- Python 3.7+
- OpenCV 4.0+
- ONNX Runtime (for FaceNet model)

### Python Dependencies
```
mtcnn>=0.1.0
numpy>=1.16.4
scipy>=1.3.0
opencv-python>=4.0.0
matplotlib>=3.0.0
scikit-learn>=0.22.0
```

### Additional Files
- `resnet50_128.onnx`: Pre-trained FaceNet model for face embeddings
- Training datasets organized in folders by person name

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/miteshkumar07/Face_Recognition_-_Detection.git
   cd Face-Recognition-and-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the FaceNet model**
   - Place `resnet50_128.onnx` in the project root directory
   - This model provides 128-dimensional face embeddings

## Project Structure

```
Face-Recognition-and-Detection/
├── classifier.py           # k-NN classifier implementation
├── dir_curve.py           # DIR curve evaluation script
├── evaluation.py          # Open-set evaluation framework
├── face_detector.py       # Face detection and tracking
├── face_recognition.py    # Face recognition and clustering
├── test.py               # Testing/inference script
├── training.py           # Training script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── LICENSE              # GPL v3 License
└── .gitignore          # Git ignore rules
```

## Usage

### Training Mode

**For Face Recognition (Supervised)**
```bash
python training.py --mode ident --video ./datasets/person_name/%04d.jpg --label "Person Name"
```

**For Face Clustering (Unsupervised)**
```bash
python training.py --mode cluster --video ./datasets/mixed_faces/%04d.jpg
```

**Live Training with Webcam**
```bash
python training.py --mode ident --video None --label "Your Name"
```

### Testing/Inference Mode

**Face Recognition Testing**
```bash
python test.py --mode ident --video ./test_videos/test_video.mp4
```

**Face Clustering Testing**
```bash
python test.py --mode cluster --video ./test_videos/test_video.mp4
```

**Live Testing with Webcam**
```bash
python test.py --mode ident --video None
```

### Performance Evaluation

**Generate DIR Curves**
```bash
python dir_curve.py
```
*Note: Requires `evaluation_training_data.pkl` and `evaluation_test_data.pkl`*

## Configuration Parameters

### Face Detection Parameters
- `tm_window_size`: Template matching window size (default: 20)
- `tm_threshold`: Template matching confidence threshold (default: 0.7)
- `aligned_image_size`: Size of aligned face images (default: 224)

### Face Recognition Parameters
- `num_neighbours`: Number of neighbors for k-NN (default: 11)
- `max_distance`: Maximum distance threshold for unknown detection (default: 0.8)
- `min_prob`: Minimum probability threshold for identification (default: 0.5)

### Face Clustering Parameters
- `num_clusters`: Number of clusters for k-means (default: 5)
- `max_iter`: Maximum iterations for k-means (default: 25)

## Key Components

### 1. Face Detection (`face_detector.py`)
- **MTCNN-based Detection**: Robust face detection with landmark estimation
- **Template Matching Tracking**: Efficient face tracking across frames
- **Face Alignment**: Standardized face preprocessing

### 2. Face Recognition (`face_recognition.py`)
- **FaceNet Embeddings**: 128-dimensional face feature extraction
- **k-NN Classification**: Nearest neighbor-based identity prediction
- **Open-set Recognition**: Unknown face detection with confidence measures

### 3. Face Clustering (`face_recognition.py`)
- **k-means Clustering**: Unsupervised face grouping
- **Embedding-based Similarity**: Feature-space clustering
- **Re-identification**: Cluster-based face matching

### 4. Performance Evaluation (`evaluation.py`)
- **Open-set Evaluation**: Comprehensive performance assessment
- **DIR Curves**: Detection and Identification Rate analysis
- **Threshold Optimization**: Automatic threshold selection

## Performance Metrics

- **Identification Rate**: Percentage of correctly identified known faces
- **False Alarm Rate**: Percentage of unknown faces incorrectly identified
- **DIR Curves**: Trade-off visualization between identification and false alarm rates

## Controls

### During Training/Testing
- **ESC**: Exit the application
- **Spacebar**: Pause/resume (in some modes)

### Video Input Formats
- **Webcam**: Set `--video None`
- **Video File**: Provide path to video file
- **Image Sequence**: Use format like `./folder/%04d.jpg`

## Data Format

### Training Data Structure
```
datasets/
├── training_data/
│   ├── Person1/
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   ├── Person2/
│   │   ├── 0001.jpg
│   │   └── ...
│   └── ...
└── test_data/
    └── (similar structure)
```

### Saved Models
- `recognition_gallery.pkl`: Trained face recognition model
- `clustering_gallery.pkl`: Trained face clustering model
- `evaluation_training_data.pkl`: Training embeddings for evaluation
- `evaluation_test_data.pkl`: Test embeddings for evaluation

## Technical Details

### Face Embedding
- **Model**: ResNet-50 based FaceNet
- **Embedding Size**: 128 dimensions
- **Normalization**: L2 normalized embeddings
- **Input Size**: 224×224 RGB images

### Classification
- **Algorithm**: k-Nearest Neighbors (k-NN)
- **Distance Metric**: Euclidean distance
- **Open-set Strategy**: Distance and probability thresholding

### Clustering
- **Algorithm**: k-means clustering
- **Initialization**: Random centroid initialization
- **Convergence**: Centroid movement tolerance: 1e-4

## Troubleshooting

### Common Issues

1. **ONNX Model Not Found**
   - Ensure `resnet50_128.onnx` is in the project root
   - Check file permissions

2. **Poor Detection Performance**
   - Adjust `tm_threshold` parameter
   - Ensure good lighting conditions
   - Check camera resolution settings

3. **Recognition Accuracy Issues**
   - Increase training samples per person
   - Adjust `num_neighbours` parameter
   - Fine-tune distance thresholds

4. **Memory Issues**
   - Reduce `aligned_image_size`
   - Limit number of training samples
   - Use smaller batch sizes for evaluation

## Future Improvements

- Add support for multiple face detection models
- Implement deep learning-based face recognition
- Add face quality assessment
- Support for video batch processing
- Web interface for easier interaction
- Mobile app integration
- Database integration for large-scale deployment

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **MTCNN**: For robust face detection
- **FaceNet**: For face embedding extraction
- **OpenCV**: For computer vision utilities
- **scikit-learn**: For machine learning algorithms

## Contact

For questions, suggestions, or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This system is designed for research and educational purposes. For production use, please ensure compliance with privacy regulations and ethical AI guidelines.
