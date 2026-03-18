# AIgen_Image_Detection_Via_DL
## Image forensics using deep-learning based image reconstruction and pattern detection ##


Preliminary Pipeline:

### Data Source: FIDD

Contains real images and fake images generated from the following GAN/Diffusion models:

StyleGAN → periodic artifacts
Diffusion → high frequency noise residue
VAE → smooth texture fields

Base encoder: Self-supervised MAE. 
We will use the pretrained MAE backbone from "facebook/vit-mae-base". This model will be trained on the real images data from FIDD.

### Experiment 1: Unsupervised Anomaly Detection

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

### Experiment 2: CLS Token Classifier

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

### Experiment 3: Variance-based Classifier

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

### Experiment 3.5(Multi-class version of 3): Generator Attribution

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


### Tabulated Summary:

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

### Addtional Feature for V1.01: GAN(Pytorch) trained on COCO to generate GAN fake imageset with DANN later to domain-proof trained Descriminator

The Architecture Breakdown

### Step 1. GAN Generator & Descriminator:

The Generator (G): Takes a latent noise vector (or a source image) and produces a synthetic image attempting to mimic the target distribution.

The Discriminator (D): Examines an image and provides the standard adversarial GAN loss.

Save the best performing D model.

### Step 2. DANN domain-proofing Descriminator:

The Discriminator (D): Best performing Descriminator loaded from GAN model, tunes against two separate sources of fake images.

The Domain Classifier (D domain): Looks at the internal feature representations and performs domain classification.

The Gradient Reversal Layer (GRL): Maximizes gradient of the backward loop of Domain Classifier to eventually make model domain invariant.


# Report Model Performance Comparisons in Report



