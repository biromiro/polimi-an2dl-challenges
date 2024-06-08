# Time Series Forecasting Using Advanced Deep Learning Techniques

This repository contains the code and documentation for the "Time Series Forecasting" challenge, a part of the Artificial Neural Networks and Deep Learning course at Politecnico di Milano, Italy. The challenge focuses on predicting future values in time series data across various categories.

## Repository Structure

- **src/**: Contains all Jupyter notebooks with the models and experiments.
  - `transformer.ipynb`: Notebook detailing experiments with Transformer models.
  - `transformer_cat.ipynb`: Notebook with category-specific Transformer models.
  - `tsmixer.ipynb`: Experiments using the TSMixer architecture.
  - `analysis_and_basic_models.ipynb`: Initial data analysis and basic model implementations.
  - `informer.ipynb`: Experiments with the Informer model, focusing on long sequence time series forecasting.

- **docs/**: Contains the comprehensive project report and related images.
  - `report.pdf`: Detailed report describing methodologies, experiments, and findings of the project.

## Project Overview

The project undertakes the challenge of forecasting future samples in time series data, leveraging a dataset of 48,000 time series across six categories. The key techniques explored include:

1. **Data Analysis**: Analyzing the structure and properties of the time series across different categories.
2. **Preprocessing and Data Preparation**: Strategies for maximizing data utility, addressing issues like padded sequences and window sizing.
3. **Model Exploration**:
   - Bidirectional LSTM and GRU models with 1D Convolutional layers.
   - Attention mechanisms and Transformer architectures.
   - Ensemble models combining different window sizes.
   - TSMixer architecture for potentially state-of-the-art performance.
4. **Best Practices and Results**: Highlighting the best configurations and model performances based on MSE and MAE metrics.

## Best Submission

The top-performing model configuration used the TSMixer architecture, optimized with a Feed-Forward dimension of 128 and a window size of 200, achieving the most promising results locally, though with challenges in scaling performance to Codalab competitions. We came at $295^{th}$ out of $518$ submissions.

## Team Contributions

- **Francesco Caserta**: Focused on dataset splitting strategies and model experimentation.
- **Nuno Costa**: Specialized in state-of-the-art architectures and data augmentation techniques.
- **Shodai Fujimoto**: Dedicated to experimenting with various window sizes and ensemble strategies.
- **Rio Ishibashi**: Concentrated on exploring different model structures and dataset splitting approaches.

## References

Refer to the `report.pdf` in the **docs** folder for detailed references and further reading related to the methodologies and architectures used in this project.