# AIgen_Image_Detection_Via_DL
# Image forensics using deep-learning based image reconstruction and pattern detection #


Preliminary Pipeline:

# Data Source: FIDD

Contains real images and fake images generated from the following GAN/Diffusion models:

StyleGAN → periodic artifacts
Diffusion → high frequency noise residue
VAE → smooth texture fields

Base encoder: Self-supervised MAE. 
We will use the pretrained MAE backbone from "facebook/vit-mae-base". This model will be trained on the real images data from FIDD.

# Experiment 1: Unsupervised Anomaly Detection

Objective: Detect AI-gen images via per-patch reconstruction error

Pipeline:

image
 ↓
mask patches
 ↓
MAE reconstruction
 ↓
compute reconstruction error
 ↓
threshold → real / fake

Performance Metrics:

ROC-AUC
Precision
Recall
F1 Score
False Positive Rate (FPR)

Visualization:

Patch-wise reconstruction error heatmaps

# Experiment 2: CLS Token Classifier

Objective: Supervised learning for classifying real vs fake using learned features.

Pipeline:

image
 ↓
MAE encoder
 ↓
CLS token
 ↓
MLP classifier
 ↓
prediction

Variables:

Frozen encoder
Fine-tuned encoder (optional)

Performance Metrics:

Accuracy
ROC-AUC
Precision
Recall
F1 Score

Visualization:

Patch-wise reconstruction error heatmaps

# Experiment 3: Variance-based Classifier

Objective: Detect AI-generated images using statistical patterns across patch embeddings

Pipeline:

image
 ↓
MAE encoder
 ↓
patch embeddings (N × D)
 ↓
compute variance across patches
 ↓
variance vector (D)
 ↓
classifier
 ↓
prediction

Base Theory:

Real images → high patch diversity
Fake images → structured / lower diversity

Performance Metrics:

Accuracy
ROC-AUC
Precision
Recall
F1 Score

# Experiment 3.5(Multi-class version of 3): Generator Attribution

Objective: Classify each fake image as being the product of one of the following generative models.(Will start with the 3 included in FIDD, can test performance on images from unseen models later):

StyleGAN
Diffusion
VAE

Base theory:

Each generative model has their own signature that can be detected via variance statistics.

Performance Metrics/Visualization:

Accuracy
Macro F1
Confusion Matrix


# Tabulated Summary:

| Experiment | Method               | Type         | Key Strength              |
| ---------- | -------------------- | ------------ | ------------------------- |
| 1          | Reconstruction error | Unsupervised | No labels needed          |
| 2          | CLS classifier       | Supervised   | Standard baseline         |
| 3          | Variance classifier  | Supervised   | Captures patch statistics |
| 4          | Attribution          | Multiclass   | Identifies generator      |

Recommended Global Metrics(subset of tracked):

ROC-AUC(Not useful for multiclass 3.5)
F1 Score
Accuracy

# Addtional Feature for V1.01: DANN-GAN(Pytorch) trained on COCO to generate GAN fake imageset

The Architecture Breakdown
The Generator (G): This is your "artist." It takes a latent noise vector (or a source image) and produces a synthetic image attempting to mimic the target distribution.

The Discriminator (D): This is the "art critic." It examines an image and asks: "Is this a real photo or a computer-generated one?" This component provides the standard adversarial GAN loss.

The Domain Classifier (D domain): This is the "detective." It looks at the internal feature representations and asks: "Do these features look like they originated from the COCO dataset or from the Generator's synthetic distribution?"

The Gradient Reversal Layer (GRL): This is the "double agent" sitting between the Generator and the Domain Classifier.

Forward Pass: It acts as an identity transform, passing features to the classifier.

Backward Pass: It multiplies the gradients by −λ, effectively forcing the Generator to learn features that are indistinguishable between domains (Domain-Invariant).

# Report Model Performance Comparisons in Report



