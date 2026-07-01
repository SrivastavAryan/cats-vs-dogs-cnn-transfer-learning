# Cats vs Dogs Image Classification - Custom CNN vs ResNet50 Transfer Learning
Built a 4-block CNN with data augmentation, BatchNorm, GlobalAveragePooling, Dropout, and Softmax on 12,000 Cats vs Dogs images, achieving 85.9% test accuracy and 0.859 F1 score. Benchmarked against a frozen ResNet50 transfer-learning baseline and analyzed preprocessing, input-resolution, and fine-tuning limitations.
A deep learning image-classification project that compares a custom Convolutional Neural Network built from scratch against a ResNet50 transfer-learning baseline on the TensorFlow Datasets Cats vs Dogs dataset.

The project demonstrates CNN architecture design, data augmentation, Global Average Pooling, transfer learning, model evaluation, training-time comparison, and error analysis.

## Open in Google Colab

After pushing this repository to GitHub, replace `<your-github-username>` in the link below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-github-username>/cats-vs-dogs-cnn-transfer-learning/blob/main/Notebooks/cats_vs_dogs_cnn_clean.ipynb)

## Project Objective

The objective is to classify images as either `Cat` or `Dog` and compare two CNN-based approaches:

1. A custom CNN trained from scratch
2. A frozen ResNet50 transfer-learning model with a custom classification head

The project focuses on:

- CNN architecture design
- Image preprocessing and normalization
- Data augmentation
- Global Average Pooling instead of Flatten
- Custom CNN vs transfer learning comparison
- Accuracy, precision, recall, and F1-score evaluation
- Confusion matrix analysis
- Training-time and parameter-count comparison
- Practical limitations under Google Colab hardware constraints

## Repository Structure

```text
cats-vs-dogs-cnn-transfer-learning/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- Notebooks/
|   |-- cats_vs_dogs_cnn_clean.ipynb
|   |-- cats_vs_dogs_cnn_executed.ipynb
|
|-- Reports/
|   |-- Figures/
|   |   |-- sample_images.png
|   |   |-- class_distribution.png
|   |   |-- custom_cnn_results.png
|   |   |-- transfer_learning_results.png
|   |   |-- model_comparison.png
|   |   |-- training_curves_comparison.png
|   |   |-- predictions_custom_cnn.png
|   |   |-- predictions_transfer_learning.png
|   |
|   |-- Analysis/
|       |-- model_analysis.md
|
|-- Data/
    |-- README.md
```

## Dataset

The project uses the `cats_vs_dogs` dataset from TensorFlow Datasets.

Dataset details from the executed notebook:

| Item | Value |
|---|---:|
| Dataset | `cats_vs_dogs` |
| Source | TensorFlow Datasets |
| Total samples used | 12,000 |
| Training samples | 10,800 |
| Test samples | 1,200 |
| Train/test split | 90/10 |
| Number of classes | 2 |
| Image shape | 160 x 160 x 3 |
| Training class distribution | Cats: 5,393; Dogs: 5,407 |
| Test class distribution | Cats: 612; Dogs: 588 |

The dataset is not stored in this repository. It is loaded directly using `tensorflow_datasets`.

## Classes

```text
0 -> Cat
1 -> Dog
```

## Preprocessing

Images are resized to:

```text
160 x 160 x 3
```

The notebook normalizes images to the range:

```text
[0, 1]
```

For the custom CNN, this preprocessing is appropriate. For ResNet50 transfer learning, a future improvement is to use the official `tf.keras.applications.resnet50.preprocess_input` preprocessing function because ImageNet-pretrained ResNet models expect a specific input format.

## Custom CNN Architecture

The custom CNN uses four convolutional blocks followed by Global Average Pooling.

Architecture summary:

```text
Input image: 160 x 160 x 3

Data Augmentation:
- RandomFlip
- RandomRotation
- RandomZoom

CNN Blocks:
- Conv2D + BatchNormalization + MaxPooling
- Conv2D + BatchNormalization + MaxPooling
- Conv2D + BatchNormalization + MaxPooling
- Conv2D + BatchNormalization + MaxPooling

Classifier:
- GlobalAveragePooling2D
- Dropout
- Dense Softmax Output
```

Key details:

| Item | Value |
|---|---:|
| Conv blocks | 4 |
| Total parameters | 390,850 |
| Trainable parameters | 389,890 |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Epochs | 12 |
| Batch size | 32 |
| Loss | Categorical Cross-Entropy |

## Transfer Learning Architecture

The transfer-learning model uses ResNet50 as a frozen feature extractor.

Architecture summary:

```text
Input image: 160 x 160 x 3
Frozen ResNet50 base
GlobalAveragePooling2D
Dense Softmax Output
```

Key details:

| Item | Value |
|---|---:|
| Base model | ResNet50 |
| Pretrained weights | ImageNet |
| Frozen layers | 175 |
| Trainable layers | 4 |
| Total parameters | 23,591,810 |
| Trainable parameters | 4,098 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Epochs | 15 |
| Batch size | 32 |
| Loss | Categorical Cross-Entropy |

## Results

| Metric | Custom CNN | ResNet50 Transfer Learning |
|---|---:|---:|
| Accuracy | 0.8592 | 0.6167 |
| Precision | 0.8591 | 0.6214 |
| Recall | 0.8591 | 0.6143 |
| F1-Score | 0.8591 | 0.6099 |
| Training Time | 282.5 sec | 302.0 sec |
| Total Parameters | 390,850 | 23,591,810 |
| Initial Loss | 0.7432 | 0.7842 |
| Final Loss | 0.3119 | 0.6577 |
| Loss Reduction | 58.03% | 16.14% |

## Key Findings

- The custom CNN clearly outperformed the frozen ResNet50 baseline on this experiment.
- The custom CNN achieved approximately 24 percentage points higher accuracy than the ResNet50 transfer-learning model.
- Global Average Pooling helped keep the custom CNN compact while reducing overfitting risk.
- The custom CNN had around 390K parameters, while ResNet50 had over 23M total parameters.
- ResNet50 had only 4,098 trainable parameters because the base model was frozen.
- The transfer-learning result should be treated as a baseline experiment, not as proof that transfer learning is generally weak.
- A likely improvement for the ResNet50 baseline is to use ResNet-specific preprocessing and optionally fine-tune selected deeper layers.

## Visual Outputs

Recommended figures to save under `reports/figures/`:

```text
sample_images.png
class_distribution.png
custom_cnn_results.png
transfer_learning_results.png
model_comparison.png
training_curves_comparison.png
predictions_custom_cnn.png
predictions_transfer_learning.png
```

## How to Run

### Option 1: Run in Google Colab

Open the clean notebook in Google Colab using the badge at the top of this README.

The dataset will be downloaded automatically through TensorFlow Datasets.

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/<your-github-username>/cats-vs-dogs-cnn-transfer-learning.git
cd cats-vs-dogs-cnn-transfer-learning
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the clean notebook:

```bash
jupyter notebook notebooks/cats_vs_dogs_cnn_clean.ipynb
```

## Tech Stack

- Python
- TensorFlow / Keras
- TensorFlow Datasets
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

## Skills Demonstrated

- CNN architecture design
- Transfer learning
- Image preprocessing
- Data augmentation
- Global Average Pooling
- Batch normalization
- Dropout regularization
- Hyperparameter tuning
- Model comparison
- Classification metrics
- Confusion matrix analysis
- Training-time comparison
- Parameter-count analysis
- Deep learning experimentation under hardware constraints

## Important Notes Before Public GitHub Upload

Before uploading the notebook publicly, update the following:

1. Save plots to `reports/figures/` instead of the notebook root directory.
2. Fix comments that mention `224 x 224` if the actual image size is `160 x 160`.
3. Correct comments where `CNN_EPOCHS = 12` is described as 15 epochs.
4. Correct comments where `CNN_LR = 0.0005` is described as `0.0001`.
5. Fix the EarlyStopping print statement because the code uses `patience=20` but prints `patience=5`.
6. For ResNet50, consider using `tf.keras.applications.resnet50.preprocess_input`.
7. Update the final analysis text so training times and loss reduction match the actual comparison table.
8. Remove or externalize large embedded base64 screenshots from the portfolio version if the notebook becomes too large.

## Future Improvements

- Use ResNet50-specific preprocessing via `preprocess_input`
- Try image size 224 x 224 if memory allows
- Fine-tune the top ResNet blocks after training the classification head
- Compare with MobileNetV2 or EfficientNetB0 for lightweight transfer learning
- Add ROC-AUC and precision-recall analysis
- Deploy a simple image-upload classifier using Streamlit or Hugging Face Spaces

## Disclaimer

This project is for educational and portfolio purposes only.
