# PlantVillage SmolVLM-500M Android Package

## Files
- smolvlm-plantvillage-Q4_K_M.gguf  (main language model, Q4_K_M, ~303 MB)
- mmproj-smolvlm-Q8_0.gguf          (vision projector, Q8_0, ~109 MB)

## Model Info
- Base: HuggingFaceTB/SmolVLM-500M-Instruct
- Fine-tuned on PlantVillage: 38 classes, 4,104 training samples
- Validation accuracy: 91.5% disease, 98.5% plant
- Quantization: Q4_K_M (main), Q8_0 (mmproj)

## Integration Options

### Flutter (mt_llmkit package)
The mt_llmkit package supports SmolVLM directly [citation:1][citation:7].

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
      'Identify the plant disease in this image. <image>',
      images: [image],
    ).listen((chunk) { print(chunk.text); });

### Kotlin (kotlinllamacpp)
The kotlinllamacpp library supports multimodal with mmproj [citation:2].

    // Load model with projector
    llamaHelper.load(
        path = mainModelUri,
        contextLength = 4096,
        mmprojPath = mmprojUri
    ) { id -> /* Ready */ }
    
    // Run inference
    llamaHelper.predict(
        prompt = "Identify the plant disease in this image.",
        imagePath = selectedImageUri
    )

## Critical Prompt Format
Pass ONLY the raw user message:
    "Identify the plant disease in this image."

DO NOT pass ChatML markers. The library applies the model's template automatically.

## Output Format
    **Plant:** Tomato
    **Disease:** Early blight
    **Symptoms:** Visible symptoms consistent with Early blight.

## Parsing
    val disease = output.lines()
        .firstOrNull { it.contains("Disease:") }
        ?.substringAfter("Disease:")
        ?.replace("*", "")
        ?.trim()

## Performance (estimated)
- Inference: 3-8 seconds per image on mid-range Android
- RAM: ~600-800 MB during inference
- Fully offline after installation
