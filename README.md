# Low-Light Image Enhancement and Unsafe Zone Analysis Using Digital Image Processing and CNN

## Overview
This project presents a hybrid Digital Image Processing and Deep Learning framework for analyzing low-light traffic images. The system enhances dark images, segments potentially unsafe low-visibility regions, and uses a basic Convolutional Neural Network (CNN) for traffic-related scene classification. The final output is deployed through a Streamlit application for interactive demonstration.

## Objective
The main objective of this project is to improve the visibility of low-light traffic images and identify potentially unsafe regions in urban night-time scenes. The system also provides a safety score, risk level, heatmap, and recommendation for better interpretation.

## Features
- Low-light image enhancement using gamma correction, CLAHE, and denoising
- Image segmentation using K-means clustering
- Unsafe region detection using thresholding and mask generation
- Unsafe zone highlighting and heatmap visualization
- CNN-based traffic scene support classification
- Streamlit web application for interactive image upload and analysis

## Technologies Used
- Python
- OpenCV
- NumPy
- TensorFlow / Keras
- Streamlit
- Matplotlib
- Scikit-learn

## Project Workflow
1. Input low-light traffic image
2. Image enhancement
3. Segmentation
4. Unsafe region mask generation
5. Overlay and heatmap creation
6. CNN-based classification
7. Safety score and recommendation generation
8. Display through Streamlit application

## Dataset Information
This project is based on the ExDark dataset, which contains real low-light images captured under poor illumination conditions.

Due to GitHub file size limitations, the complete dataset is **not uploaded** to this repository. Instead:
- a few sample low-light images used for testing/demo are included
- the training notebook is included
- the final Streamlit app is included

Traffic-related classes such as **Car, Bus, Bicycle, and Motorbike** were used during model training.

## Files Included in This Repository
- `app.py` → Streamlit application
- `training.ipynb` → notebook used for training and experimentation
- `requirements.txt` → required Python libraries
- `settings.json` → project/editor settings
- sample low-light images (`.jpg`, `.png`) → demo/testing images

## Important Note About Model File
The trained CNN model file (`unsafe_zone_cnn_model.h5`) may not be present in the repository if it exceeds upload limits.

If it is not included, you can:
1. retrain the model using `training.ipynb`
2. save the trained model as `unsafe_zone_cnn_model.h5`
3. place it in the same folder as `app.py`

## How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/your-username/low-light-unsafe-zone-analysis.git
cd low-light-unsafe-zone-analysis
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Make sure the trained model file is available

Place `unsafe_zone_cnn_model.h5` in the project root folder.

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

If `streamlit` is not recognized, use:

```bash
python -m streamlit run app.py
```

## Application Outputs

The Streamlit application displays:

* Original Image
* Enhanced Image
* Final Optimized Output
* Unsafe Region Mask
* Unsafe Regions Highlighted
* Risk Heatmap
* CNN Support Classification
* CNN Confidence
* Safety Score
* Risk Level
* Recommendation

## Academic Relevance

From the Digital Image Processing syllabus, this project uses:

* Image enhancement
* Intensity transformation
* Contrast enhancement
* Filtering
* Grayscale conversion
* Thresholding
* Image segmentation
* Morphological operations
* Region-based analysis

It also includes a basic CNN as the deep learning component.

## Limitations

* The CNN model shows signs of overfitting
* Unsafe region detection is mainly based on visibility and intensity
* The CNN is used as a support classification module, not full semantic scene understanding
* The full ExDark dataset is not stored in this repository due to GitHub upload limitations

## Future Scope

* Use semantic segmentation models such as U-Net
* Improve generalization with data augmentation
* Add object detection for more detailed scene analysis
* Improve risk estimation using scene semantics

## Author

**Khushi Wadhawan**
* B.Tech CSE
* Digital Image Processing Project

## License

This project is created for academic purposes only.

```


