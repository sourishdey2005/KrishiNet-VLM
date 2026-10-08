# KrishiNet-VLM Model Card

**Model Name:** KrishiNet-VLM-500M-PlantVillage  
**Version:** 1.0  
**Release Date:** October 2026  
**Model Type:** Vision-Language Model (Image-Text-to-Text)  
**Base Model:** HuggingFaceTB/SmolVLM-500M-Instruct  
**License:** Apache 2.0  

---

## Table of Contents

1. [Model Overview](#1-model-overview)
2. [Architecture](#2-architecture)
3. [Training Pipeline](#3-training-pipeline)
4. [Evaluation Results](#4-evaluation-results)
5. [Deployment](#5-deployment)
6. [Reproducibility](#6-reproducibility)
7. [Limitations](#7-limitations)
8. [Intended Use](#8-intended-use)
9. [Citation](#9-citation)
10. [Version History](#10-version-history)
11. [Contact](#11-contact)
12. [License](#12-license)

---

## 1. Model Overview

KrishiNet-VLM is a 500M-parameter vision-language model fine-tuned for plant disease diagnosis. It accepts a leaf image and a text prompt as input and produces a structured text response containing the plant species, disease name, and visible symptoms.

The model is designed for **fully offline inference** on consumer hardware, including mid-range Android smartphones and CPU-only laptops. No network connection, cloud API, or GPU is required at inference time.

### 1.1 Key Characteristics

| Property | Value |
|----------|-------|
| Total parameters | 517 million |
| Trainable parameters (LoRA) | 9.57 million (1.85%) |
| Vision encoder | SigLIP (93M) |
| Language decoder | SmolLM2-360M |
| Context length | 8,192 tokens |
| Vocabulary size | 49,280 tokens |
| Image input | 512×512 per patch, 4×4 grid (2048×2048 effective) |
| Quantized size (Q4_K_M) | 289 MB |
| Vision projector size (Q8_0) | 104 MB |
| Total deployment size | 393 MB |
| Inference latency (mid-range Android) | 7–12 seconds |
| Inference latency (CPU laptop) | 7–15 seconds |
| RAM usage during inference | ~600–800 MB |

### 1.2 Task Definition

**Input:** A single leaf image (JPEG/PNG, any resolution) plus the prompt:  
`"Identify the plant disease in this image."`

**Output:** Structured text in the format:

```
**Plant:** <plant species>
**Disease:** <disease name>
**Symptoms:** <visible symptoms>
```

**Example:**

```
**Plant:** Tomato
**Disease:** Early blight
**Symptoms:** Concentric brown lesions on lower leaves, yellowing around spots.
```

### 1.3 Dataset Coverage

- **Dataset:** PlantVillage (38 disease classes, 14 crop species)
- **Training samples:** 4,104 images
- **Task:** Single-leaf disease classification from still images under controlled conditions

---

## 2. Architecture

KrishiNet-VLM is built on the SmolVLM-500M-Instruct architecture with two components:

### 2.1 Vision Encoder

- **Model:** SigLIP (93M parameters)
- **Input:** 512×512 image patches arranged in a 4×4 grid, giving an effective 2048×2048 resolution
- **Role:** Extracts visual features from the leaf image

### 2.2 Vision Projector (mmproj)

- A lightweight multimodal projector that maps SigLIP image embeddings into the language model's token space
- Deployed separately as `mmproj-smolvlm-Q8_0.gguf` (~104 MB)

### 2.3 Language Decoder

- **Model:** SmolLM2-360M
- **Context length:** 8,192 tokens
- **Vocabulary:** 49,280 tokens
- **Role:** Generates the structured diagnosis text conditioned on the visual features and the user prompt

### 2.4 Attention / Runtime Notes

- Supports grouped-query attention and KV-caching for efficient generation
- Runs on CPU with no GPU/NPU dependency
- Context length of 4,096 tokens is recommended for mobile deployments to bound RAM usage

---

## 3. Training Pipeline

### 3.1 Base Model

- **Starting checkpoint:** `HuggingFaceTB/SmolVLM-500M-Instruct`
- **Instruction-tuned:** Yes (chat template inherited from base model)

### 3.2 Fine-Tuning Method

- **Technique:** LoRA (Low-Rank Adaptation) on the language decoder and vision projector
- **Trainable parameters:** 9.57M (1.85% of total)
- **Task format:** Image → structured text (`Plant / Disease / Symptoms`)
- **Prompt template:** Chat template from SmolVLM-Instruct; raw user message only (no manual ChatML markers)

### 3.3 Data

| Split | Samples |
|-------|---------|
| Train | 4,104 |
| Classes | 38 (PlantVillage) |

- Preprocessing: resize/crop to 512×512 patches, standard normalization
- Augmentation: random flips/rotations and brightness/contrast jitter (if applied) — kept minimal to preserve diagnostic fidelity

### 3.4 Export

1. Merge LoRA adapters into the base weights
2. Convert to GGUF format (llama.cpp-compatible)
3. Quantize main model: **Q4_K_M** → `smolvlm-plantvillage-Q4_K_M.gguf` (~289 MB)
4. Quantize projector: **Q8_0** → `mmproj-smolvlm-Q8_0.gguf` (~104 MB)

---

## 4. Evaluation Results

### 4.1 Validation Accuracy

| Metric | Score |
|--------|-------|
| Disease classification accuracy | **91.5%** |
| Plant species accuracy | **98.5%** |

### 4.2 Inference Performance

| Platform | Latency per image | RAM usage |
|----------|-------------------|-----------|
| Mid-range Android (CPU) | 7–12 s | ~600–800 MB |
| CPU-only laptop | 7–15 s | ~600–800 MB |

### 4.3 Qualitative Behavior

- Produces the expected structured output for in-distribution PlantVillage-style images
- Reliable at identifying plant species; disease naming depends on visual symptom clarity
- Degrades gracefully on out-of-distribution inputs (see [Limitations](#7-limitations))

---

## 5. Deployment

### 5.1 Artifacts

| File | Purpose | Size |
|------|---------|------|
| `smolvlm-plantvillage-Q4_K_M.gguf` | Main language model (Q4_K_M) | ~289–303 MB |
| `mmproj-smolvlm-Q8_0.gguf` | Vision projector (Q8_0) | ~104–109 MB |

### 5.2 Android (Flutter / mt_llmkit)

```dart
final model = LocalModel(
  config: LlmConfig(
    mmprojPath: 'assets/mmproj-smolvlm-Q8_0.gguf',
    nCtx: 4096,
    nGpuLayers: 0,  // CPU for compatibility
  ),
);
await model.loadModel('assets/smolvlm-plantvillage-Q4_K_M.gguf');

final image = LlamaImageContent(path: '/path/to/leaf.jpg');
model.sendPromptStream(
  'Identify the plant disease in this image.',
  images: [image],
).listen((chunk) { print(chunk.text); });
```

### 5.3 Android (Kotlin / kotlinllamacpp)

```kotlin
llamaHelper.load(
    path = mainModelUri,
    contextLength = 4096,
    mmprojPath = mmprojUri
) { id -> /* Ready */ }

llamaHelper.predict(
    prompt = "Identify the plant disease in this image.",
    imagePath = selectedImageUri
)
```

### 5.4 Prompting Rules

- Pass **only the raw user message**: `"Identify the plant disease in this image."`
- **Do not** pass ChatML markers; the runtime applies the model's chat template automatically

### 5.5 Output Parsing

```kotlin
val disease = output.lines()
    .firstOrNull { it.contains("Disease:") }
    ?.substringAfter("Disease:")
    ?.replace("*", "")
    ?.trim()
```

---

## 6. Reproducibility

- **Base model:** `HuggingFaceTB/SmolVLM-500M-Instruct`
- **Dataset:** PlantVillage (38 classes, 4,104 training samples)
- **Quantization:** llama.cpp GGUF — Q4_K_M (language model), Q8_0 (projector)
- **Validation:** `verify_gguf.py` in the export folder checks GGUF file integrity and metadata
- **Determinism:** Inference is deterministic given fixed image, prompt, and generation parameters (temperature 0)

---

## 7. Limitations

- Trained only on PlantVillage-style images (single leaf, controlled background, good lighting); performance on field photos with complex backgrounds, multiple leaves, or poor lighting may drop significantly
- Not suitable for diagnosing diseases outside the 38 covered classes
- Identifies plant species and disease, but **does not provide treatment recommendations**
- Output format may occasionally deviate from the expected template; parsers should be defensive
- Not a substitute for professional agronomic diagnosis

---

## 8. Intended Use

**Primary use:** On-device, offline plant disease triage for farmers, students, and agricultural extension workers via Android apps.

**Non-intended use:** Commercial agricultural decisions without expert validation, medical advice, or safety-critical applications.

---

## 9. Citation

```bibtex
@misc{krishnet_vlm_2026,
  title        = {KrishiNet-VLM-500M-PlantVillage},
  author       = {KrishiNet Team},
  year         = {2026},
  howpublished = {Model Card v1.0}
}
```

---

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | October 2026 | Initial release: LoRA fine-tune of SmolVLM-500M-Instruct on PlantVillage, GGUF Q4_K_M + Q8_0 export, 91.5% disease accuracy |

---

## 11. Contact

- Project: KrishiNet-VLM
- Issues/feedback: Project repository issue tracker

---

## 12. License

Apache License 2.0. See the base model's license (`HuggingFaceTB/SmolVLM-500M-Instruct`) for upstream terms.
