# Development Log: Item Matching Module Evolution

## Phase 1: Early Attempts

### Color Mean Matching
- **Approach**: Average color of entire image
- **Problem**: Heavily affected by background/lighting interference
- **Conclusion**: Abandoned, can only be used for rough filtering

### Color Histogram/Dominant Color Ratio
- **Approach**: HSV histogram or clustering to extract dominant colors
- **Advantages**: Can distinguish light green/purple/blue
- **Problem**: Background still has significant impact
- **Conclusion**: Retained as auxiliary signal

### ORB + RANSAC Feature Point Matching
- **Approach**: Corner point matching + geometric verification
- **Problem**: Packaging has few textures, insufficient feature points; completely fails under severe angles/occlusion
- **Conclusion**: Poor performance in this scenario, abandoned

### OCR/Barcode Recognition
- **Approach**: Direct recognition of packaging text
- **Problem**: Video frames often blurry/occluded
- **Conclusion**: Only used as fallback, not main pipeline

---

## Phase 2: Semantic and Deep Feature Exploration

### Semantic Color Classification
- HSV clustering + semantic labels (light green/purple/blue/pink)
- Used for quickly excluding non-matching candidates

### Packaging Type Classification (Soft bag vs Hard box)
- Utilizes aspect ratio, edge features, etc.
- Used as constraint conditions

### Key Element Detection
- Template or shape matching for clouds/brand logos, etc.
- Distinguishes similar-colored SKUs

### Deep Features (Image Fingerprints)
- MobileNet/CLIP embedding, vector similarity
- Significantly outperforms traditional feature points, strong robustness
- **Conclusion**: Set as main approach

### LoFTR Feature Matcher (Failed Attempt)
- **Approach**: Local features + Transformer fusion, high precision
- **Problem**: Cannot run on edge devices (Raspberry Pi, etc.), excessive computational requirements
- **Conclusion**: Abandoned for edge deployment

---

## Phase 3: Multi-Signal Fusion (Current fusion_matcher.py Architecture)

### Signal Composition
- **Primary Signal**: CLIP semantic features (0.6)
- **Auxiliary Signal**: Lab color features (0.1)
- **Strong Constraint**: Material classification (0.3)

### Pipeline
1. Pre-compute inventory image features (CLIP, color, material)
2. Extract features from video frames → three types of similarity matrices
3. Construct candidate × signal matrix
4. Weighted fusion → score ranking
5. Return best match

### Characteristics
- **Completeness**: All three signals computed for each candidate
- **Adjustability**: Flexible weight adjustment
- **Interpretability**: Output individual scores for easy debugging

---

## Current Conclusions and Future Plans

### Current Status
Stop further research on model-side modifications, enter scenario testing phase

### Short-term Goals
Increase testing in real environments (different lighting, angles, occlusion)

### Long-term Plans
If edge-side fusion approach still performs poorly after large-scale testing → migrate to cloud, directly call third-party APIs (such as product recognition/image retrieval services), run heavier models in the cloud