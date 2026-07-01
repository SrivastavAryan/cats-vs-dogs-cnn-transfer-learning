# Model Analysis - Cats vs Dogs CNN Classification

## 1. Project Summary

This project compares a custom CNN trained from scratch against a frozen ResNet50 transfer-learning baseline for binary image classification on the Cats vs Dogs dataset.

The experiment uses:

- 12,000 total images
- 10,800 training images
- 1,200 test images
- 2 balanced classes: Cat and Dog
- 160 x 160 x 3 image resolution
- Accuracy as the primary metric because the dataset is approximately balanced

## 2. Dataset Balance

The train/test split is well balanced.

Training set:

| Class | Count |
|---|---:|
| Cat | 5,393 |
| Dog | 5,407 |

Test set:

| Class | Count |
|---|---:|
| Cat | 612 |
| Dog | 588 |

Because the class distribution is nearly equal, accuracy is a reasonable primary metric. Macro precision, recall, and F1-score are still reported to ensure that both classes are evaluated fairly.

## 3. Custom CNN Architecture

The custom CNN contains:

- Data augmentation layers
- Four convolutional blocks
- Batch normalization
- Max pooling
- Global Average Pooling
- Dropout
- Dense softmax output layer

Architecture pattern:

```text
Input
-> RandomFlip / RandomRotation / RandomZoom
-> Conv2D + BatchNorm + MaxPool
-> Conv2D + BatchNorm + MaxPool
-> Conv2D + BatchNorm + MaxPool
-> Conv2D + BatchNorm + MaxPool
-> GlobalAveragePooling2D
-> Dropout
-> Dense Softmax
```

Parameter count:

| Parameter Type | Count |
|---|---:|
| Total parameters | 390,850 |
| Trainable parameters | 389,890 |
| Non-trainable parameters | 960 |

The model uses Global Average Pooling instead of Flatten, which reduces parameter count and helps limit overfitting.

## 4. ResNet50 Transfer Learning Architecture

The transfer-learning model uses ResNet50 pretrained on ImageNet.

Architecture pattern:

```text
Input
-> Frozen ResNet50 base
-> GlobalAveragePooling2D
-> Dense Softmax
```

Parameter count:

| Parameter Type | Count |
|---|---:|
| Total parameters | 23,591,810 |
| Trainable parameters | 4,098 |
| Frozen layers | 175 |
| Trainable layers | 4 |

The ResNet50 base is frozen, so only the classification head is trained.

## 5. Performance Comparison

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

## 6. Confusion Matrix Interpretation

The custom CNN produced balanced performance across both classes:

- Cat precision/recall around 0.86
- Dog precision/recall around 0.86
- Macro F1 around 0.859

The ResNet50 transfer-learning baseline was weaker:

- Cat recall was higher than dog recall
- Dog classification was comparatively weaker
- Macro F1 was around 0.610

This suggests that the custom CNN learned dataset-specific visual features more effectively in the current setup.

## 7. Why the Custom CNN Performed Better

The custom CNN performed better in this experiment for several likely reasons:

1. It was trained directly on the target image distribution.
2. It used data augmentation suited to the Cats vs Dogs task.
3. It had enough capacity for binary classification without excessive complexity.
4. It used Global Average Pooling to reduce overfitting.
5. The ResNet50 baseline was frozen, allowing only the small classification head to adapt.
6. The ResNet50 input preprocessing may not have fully matched ImageNet preprocessing expectations.

## 8. Transfer Learning Limitation

The ResNet50 result should be interpreted as a baseline, not as proof that transfer learning is ineffective.

Potential reasons for weaker transfer-learning performance:

- Input size was 160 x 160 instead of the common 224 x 224 ResNet setup.
- Images were normalized to [0, 1], while ResNet50 usually benefits from `tf.keras.applications.resnet50.preprocess_input`.
- The base model was fully frozen, so deeper layers could not adapt to the Cats vs Dogs domain.
- Only 4,098 parameters were trainable, limiting adaptation capacity.

A stronger transfer-learning experiment would use ResNet-specific preprocessing, train the classification head first, then unfreeze and fine-tune selected top ResNet blocks with a low learning rate.

## 9. Global Average Pooling Impact

Global Average Pooling was used in both models.

Benefits:

- Reduces parameter count compared to Flatten
- Helps control overfitting
- Preserves spatial feature learning
- Makes the classifier head lightweight
- Aligns with modern CNN architecture practice

In the custom CNN, GAP helped produce strong performance with only 390,850 parameters.

## 10. Training and Convergence

Custom CNN:

- Initial loss: 0.7432
- Final loss: 0.3119
- Loss reduction: 58.03%

ResNet50 transfer learning:

- Initial loss: 0.7842
- Final loss: 0.6577
- Loss reduction: 16.14%

The custom CNN showed much stronger convergence. The transfer-learning model improved, but the loss reduction was modest.

## 11. Computational Cost

Although ResNet50 has far more total parameters, most of them were frozen. Therefore, training time was not dramatically worse than the custom CNN.

| Model | Training Time | Total Parameters | Trainable Parameters |
|---|---:|---:|---:|
| Custom CNN | 282.5 sec | 390,850 | 389,890 |
| ResNet50 Transfer | 302.0 sec | 23,591,810 | 4,098 |

The custom CNN offered better accuracy with far fewer total parameters.

## 12. Recommended Improvements

For future improvement:

1. Use `tf.keras.applications.resnet50.preprocess_input` for ResNet50.
2. Try 224 x 224 input size if memory allows.
3. Unfreeze the top ResNet blocks after training the head.
4. Use a smaller transfer-learning model such as MobileNetV2 or EfficientNetB0.
5. Save all figures under `reports/figures/`.
6. Add ROC-AUC and precision-recall analysis.
7. Add Grad-CAM visualization to interpret image regions used by the CNN.

## 13. Final Conclusion

The custom CNN was the best-performing model in this experiment, achieving 85.92% test accuracy and 0.8591 macro F1 score. The ResNet50 transfer-learning baseline underperformed at 61.67% accuracy, but this result is best interpreted as a useful diagnostic experiment showing the importance of preprocessing, input resolution, and fine-tuning strategy in transfer learning.

For a portfolio project, the strongest takeaway is not that transfer learning failed; it is that the project includes controlled model comparison, architecture design, metric-based evaluation, and practical debugging of deep-learning constraints.
