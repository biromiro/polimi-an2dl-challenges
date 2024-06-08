# Leaf Health Classification Using Deep Learning

This repository contains the code and documentation for the "Leaf Health Classification" challenge, which is a part of the Artificial Neural Networks and Deep Learning course at Politecnico di Milano, Italy. The project aims to classify 96x96 pixel images of leaves into healthy and unhealthy categories.

## Repository Structure

- **src/**: Contains all source code files, including Jupyter notebooks and Python scripts.
  - `outlier_detection.ipynb`: Notebook for detecting outliers in the dataset.
  - `preprocessing_viz.ipynb`: Notebook for preprocessing and visualizing data.
  - `FinalNotebook.ipynb`: Comprehensive notebook including all analyses and model training.
  - `ensemble+tta_model.py`: Python script for the ensemble model with test-time augmentation.

- **docs/**: Contains the project report and related images.
  - `report.pdf`: Detailed report describing the methodologies and findings of the project.
  - **images/**: Folder containing images used in the report and notebooks.

## Project Overview

The project involves analyzing leaf images to distinguish between healthy and unhealthy leaves using various deep learning techniques. Key components of the project include:

1. **Data Analysis**: Initial exploration of the dataset to identify duplicates and analyze image properties.
2. **Data Preprocessing and Augmentation**: Techniques such as random flipping, translation, rotation, and zooming to enhance the dataset.
3. **Model Architectures**: Implementation of several CNN architectures, including Basic CNN, MobileNetV2, EfficientNet, and ConvNeXt.
4. **Test-time Augmentation (TTA)**: Improving model performance by averaging predictions from multiple augmented images of the same input.
5. **Ensemble Modeling**: Combining predictions from several models to improve accuracy.

## Best Submission

The best results were achieved using an ensemble of three models, combining different augmentation strategies and fine-tuning approaches, with a final test accuracy of 86.6% on the Codalab platform, coming at $34^{th}$ place out of $211$ submissions.

## Contributors

- **Francesco Caserta**: Focused on model training and test-time augmentation.
- **Nuno Costa**: Led the data analysis and preprocessing, and contributed to the ensemble approach.
- **Shodai Fujimoto**: Specialized in model training with various architectures and hyperparameters.
- **Rio Ishibashi**: Worked on model explainability and implemented additional training approaches.

## References

For detailed references, please refer to the `report.pdf` in the **docs** folder.