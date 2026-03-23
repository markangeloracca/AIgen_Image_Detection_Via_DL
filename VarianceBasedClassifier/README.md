# Variance-based Classifier

## Objective:

Detect AI-generated images using statistical patterns across patch embeddings

## Pipeline

- image ↓
- Pretrained MAE encoder ↓
- patch embeddings (N × D) ↓
- compute variance across patches ↓
- variance vector (size D) ↓
- classifier ↓
- prediction

### N ,D

- N depends on image size and patch size (model config)
  - depends on image dimension
  - and patch size (fixed by encoder)
- D depends only on the MAE encoder architecture
  - D = embedding dimension

## Base Theory:

Real images

- Natural scenes → irregular textures, lighting, noise
- → higher variance across patches

AI-generated images

- Often overly smooth or structurally consistent
- → lower or more patterned variance

## Performance Metrics:

- Accuracy
- ROC-AUC
- Precision
- Recall
- F1 Score

## Limiations

1. High-quality AI images
1. Highly structured real images
1. Compression / resizing
