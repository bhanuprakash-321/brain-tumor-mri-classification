## Brain Tumor Classification Using Deep Learning
This project focuses on classifying brain tumors from MRI scans using deep learning, implementing both a custom Convolutional Neural Network (CNN) and transfer learning with a fine-tuned VGG16 model. The goal is to accurately detect and classify brain tumors into four distinct categories, aiding in early diagnosis and treatment planning.  

## Dataset


The dataset used is the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from **Kaggle**, consisting of labeled MRI images categorized into four classes:

- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**                      

Images are preprocessed (resized, normalized, and augmented) to ensure optimal training performance.

## Models and Methodology
**1. Custom CNN**
- A convolutional neural network was built from scratch.

- Includes multiple convolution, pooling, and dense layers.

- Data augmentation (rotation, flipping, zoom, etc.) was applied to prevent overfitting and improve generalization.

**2. Transfer Learning with VGG16**
- A pre-trained VGG16 model (trained on ImageNet) was used as a fixed feature extractor initially (convolutional base frozen).

- Custom dense layers were added on top to adapt the model to the brain tumor dataset.

**VGG16 Architecture Overview**

This diagram shows the layer structure of the VGG16 model, including its 13 convolutional and 3 fully connected layers.
![image](https://github.com/user-attachments/assets/fddc22cb-df95-4fce-a81c-2354bdbc64fd)


**3. Fine-Tuning VGG16**
- To improve performance, the top 15 layers of VGG16 were unfrozen.

- This allows the model to adapt high-level features specific to brain tumor classification.

- A lower learning rate was used to fine-tune without disturbing the pre-trained weights too drastically.

## Evaluation Metrics
Model performance was evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

These metrics were used on validation and test sets to ensure both performance and generalization across all tumor classes.

## Results Summary

| Model               | Validation Accuracy | Best Accuracy | Notes                                |
|--------------------|---------------------|---------------|--------------------------------------|
| Custom CNN          | ~91%                | 91%           | Good generalization with augmentation |
| VGG16 (frozen)      | ~92%                | 92%           | Strong performance out-of-the-box     |
| Fine-tuned VGG16    | **94.3%**           | **94.3%**     | Best results with fine-tuning         |

> Fine-tuning significantly improved classification accuracy by allowing the model to learn task-specific features in deeper layers.


## Conclusion
This project demonstrates the effectiveness of deep learning techniques, particularly transfer learning, for medical image classification. While the custom CNN achieved solid performance, the fine-tuned VGG16 model surpassed it, highlighting the advantage of leveraging pre-trained models and adapting them to domain-specific tasks. With further refinements, such as experimenting with more advanced architectures or ensemble methods, this approach can serve as a valuable aid in medical diagnosis and decision-making.

