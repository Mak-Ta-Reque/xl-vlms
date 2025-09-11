# GLIMPSE Implementation Guide

## Overview

This document provides a comprehensive understanding of the GLIMPSE (Gradient-weighted Layer-wise Importance for Multimodal Prompt-guided Saliency Explanation) implementation for Vision-Language Models (VLMs).

## Complete System Architecture

### Inputs
1. **Vision-Language Model**: Any autoregressive VLM (Qwen2.5-VL, LLaVA, IDEFICS2, etc.)
2. **Image**: RGB image input (processed into visual tokens)  
3. **Text Prompt**: Natural language instruction/query
4. **Target Token**: Generated token to explain (typically last generated token)

### Outputs
1. **Visual Saliency Map**: Spatial heatmap highlighting important image regions
2. **Prompt Saliency Map**: Token-level importance scores for input prompt
3. **Cross-Modal Token Relevance**: Scores for each generated token's multimodal alignment
4. **Relevance Matrix**: Full token-to-token interaction matrix

## Implementation Logic and Steps

### Stage 1: Layer Relevance Extraction

**Purpose**: Extract gradient-weighted attention for each layer, fusing across attention heads.

**Logic Flow**:
```python
# For each layer l and head h:
1. Capture attention weights A^l_h during forward pass
2. Compute gradients ∂z_t/∂A^l_h during backward pass  
3. Apply element-wise multiplication: G^l_h = ReLU(∂z_t/∂A^l_h) ⊙ A^l_h
4. Compute head importance: w^l_h = E[A^l_h | positive gradients]
5. Fuse across heads: E^l = Σ_h w^l_h * G^l_h
6. Row-normalize E^l to preserve probability mass
```

**Key Equations**:
- Eq. 5: `G^l_h = ReLU(g^l_h ⊙ A^l_h)` (gradient-weighted attention)
- Eq. 6-7: `w^l_h = softmax(E[A^l_h under positive gradients] / λ)` (head weights)
- Eq. 8: `E^l = Σ_h w^l_h * G^l_h` (fused layer attention)

### Stage 2: Adaptive Layer Propagation

**Purpose**: Propagate relevance through layers using gradient magnitude and depth priors.

**Logic Flow**:
```python
# Compute layer weights:
1. Calculate gradient L1 norm for each layer: g^l = ||Σ_h g^l_h||_1
2. Apply depth-based prior: s^l = exp(λ_d * (l+1)) / Σ_k exp(λ_d * (k+1))
3. Combine: α^l = (g^l * s^l) / Σ_k (g^k * s^k)

# Propagate relevance:
4. Initialize: R = Identity_matrix(N×N)
5. For each layer l:
   - Construct: L^l = I + α^l * E^l  
   - Update: R = R + L^l @ R (additive accumulation)
```

**Key Equations**:
- Eq. 9: `g^l = ||Σ_h g^l_h||_1` (layer gradient norm)
- Eq. 10: `s^l = exp(λ_d * (l+1)) / normalization` (depth prior)
- Eq. 11: `α^l = (g^l * s^l) / Σ_k (g^k * s^k)` (adaptive weights)
- Eq. 12-14: Relevance propagation with additive accumulation

### Stage 3: Cross-Modal Token Relevancy

**Purpose**: Compute token-level alignment and confidence weights for saliency aggregation.

**Logic Flow**:
```python
# For each generated token t:
1. Prompt alignment: a_t = mean(R[t, prompt_indices])
2. Visual alignment: v_t = mean(R[t, visual_indices])  
3. Confidence: p_t = softmax(logits[t])

# Compute token weights for each modality m:
4. For visual saliency: β_t^V = (p_t * a_t) / Σ_k (p_k * a_k)
5. For prompt saliency: β_t^P = (p_t * v_t) / Σ_k (p_k * v_k)
6. Cross-modal relevance: γ_t = β_t^V * β_t^P
```

**Key Equations**:
- Eq. 15: `a_t = (1/|P|) * Σ_{i∈P} R(t,i)` (prompt alignment)
- Eq. 16: `v_t = (1/|V|) * Σ_{i∈V} R(t,i)` (visual alignment)
- Eq. 17: `p_t = softmax(z_t)` (confidence weight)
- Eq. 19: `β_t^(m) = (p_t * w_t^(m)) / Σ_k (p_k * w_k^(m))` (final token weights)
- Eq. 20: `γ_t = β_t^(V) × β_t^(P)` (joint relevance)

### Stage 4: Holistic Saliency Aggregation

**Purpose**: Aggregate token-level explanations into unified saliency maps.

**Logic Flow**:
```python
# Aggregate across generated tokens:
1. Visual saliency: R̃_V = Σ_t β_t^V * R(t, visual_indices)
2. Prompt saliency: R̃_P = Σ_t β_t^P * R(t, prompt_indices)
3. Return spatial heatmap for image and token importance for prompt
```

**Key Equation**:
- Eq. 23: `R̃_m = Σ_{t∈Y} β_t^(m) * R(t,m)` (holistic aggregation)

## Complete Pseudocode

```python
def GLIMPSE_EXPLAIN(model, input_ids, attention_mask, pixel_values, target_token):
    """
    Complete GLIMPSE explanation pipeline
    
    INPUTS:
    - model: Vision-language transformer model
    - input_ids: Tokenized sequence [visual_tokens || prompt_tokens || generated_tokens]
    - attention_mask: Sequence attention mask
    - pixel_values: Image pixel values
    - target_token: Index of token to explain
    
    OUTPUTS:
    - visual_saliency: Spatial importance heatmap over image patches
    - prompt_saliency: Token importance scores for input prompt
    - cross_modal_relevance: Joint alignment scores for generated tokens
    - relevance_matrix: Full token-to-token relevance matrix
    """
    
    # === STAGE 1: LAYER RELEVANCE EXTRACTION ===
    
    # 1.1: Register hooks to capture attention and gradients
    attention_weights = {}  # A^l_h matrices
    attention_gradients = {}  # ∂z_t/∂A^l_h gradients
    register_attention_hooks(model, attention_weights, attention_gradients)
    
    # 1.2: Forward pass (captures attention weights)
    outputs = model(input_ids, attention_mask, pixel_values)
    logits = outputs.logits
    
    # 1.3: Backward pass (captures gradients)
    target_logit = logits[target_token]
    target_logit.backward(retain_graph=True)
    
    # 1.4: Compute layer relevance matrices
    layer_relevance_matrices = []
    FOR layer_l in range(num_layers):
        
        # For each head in layer
        gradient_weighted_attentions = []
        head_importance_weights = []
        
        FOR head_h in layer_l:
            A_lh = attention_weights[layer_l][head_h]  # Attention matrix
            g_lh = attention_gradients[layer_l][head_h]  # Gradient matrix
            
            # Eq. 5: Gradient-weighted attention
            G_lh = ReLU(g_lh) ⊙ A_lh
            gradient_weighted_attentions.append(G_lh)
            
            # Eq. 6-7: Head importance (expectation under positive gradients)
            numerator = sum(G_lh[i,j] for all i,j)
            denominator = sum(ReLU(g_lh[i,j]) for all i,j)
            head_weight = numerator / denominator
            head_importance_weights.append(head_weight)
        
        # Eq. 6: Normalize head weights with temperature
        w_lh = softmax(head_importance_weights / λ_head)
        
        # Eq. 8: Fused attention matrix
        E_l = sum(w_lh[h] * G_lh[h] for h in heads)
        E_l = row_normalize(E_l)  # Preserve probability mass
        layer_relevance_matrices.append(E_l)
    
    # === STAGE 2: ADAPTIVE LAYER PROPAGATION ===
    
    # 2.1: Compute adaptive layer weights
    gradient_norms = []
    FOR layer_l in range(num_layers):
        # Eq. 9: L1 norm of aggregated gradients
        g_l = L1_norm(sum(g_lh for head_h in layer_l))
        gradient_norms.append(g_l)
    
    # Eq. 10: Depth-based prior (favor deeper layers)
    depth_weights = []
    FOR layer_l in range(num_layers):
        s_l = exp(λ_depth * (l + 1))
        depth_weights.append(s_l)
    depth_weights = normalize(depth_weights)
    
    # Eq. 11: Combined adaptive weights
    α_l = (gradient_norms[l] * depth_weights[l]) / sum(gradient_norms * depth_weights)
    
    # 2.2: Relevance propagation through layers
    # Eq. 12: Initialize as identity matrix
    R = Identity_matrix(sequence_length, sequence_length)
    
    FOR layer_l in range(num_layers):
        E_l = layer_relevance_matrices[layer_l]
        
        # Eq. 13: Layer-specific transformation
        L_l = Identity_matrix + α_l * E_l
        
        # Eq. 14: Additive accumulation (more stable than full product)
        R = R + L_l @ R
    
    # === STAGE 3: CROSS-MODAL TOKEN RELEVANCY ===
    
    # 3.1: Define token index sets
    V = {0, ..., K-1}        # Visual token indices
    P = {K, ..., K+M-1}      # Prompt token indices  
    Y = {K+M, ..., K+M+T-1}  # Generated token indices
    
    # 3.2: Compute cross-modal weights for each generated token
    cross_modal_weights = {}
    FOR token_t in generated_tokens:
        # Eq. 15: Prompt alignment weight
        a_t = mean(R[t, i] for i in P)
        
        # Eq. 16: Visual alignment weight
        v_t = mean(R[t, i] for i in V)
        
        # Eq. 17: Confidence weight (model certainty)
        p_t = softmax(logits[t])[predicted_token_t]
        
        cross_modal_weights[token_t] = {
            'prompt_alignment': a_t,
            'visual_alignment': v_t,
            'confidence': p_t
        }
    
    # 3.3: Compute token weights for saliency aggregation
    # Eq. 18-19: Combined weighting for each modality
    FOR modality m in [V, P]:
        FOR token_t in generated_tokens:
            IF modality == V:  # Visual saliency
                # Use prompt alignment for visual saliency
                w_t = cross_modal_weights[token_t]['prompt_alignment']
            ELSE:  # Prompt saliency
                # Use visual alignment for prompt saliency
                w_t = cross_modal_weights[token_t]['visual_alignment']
            
            # Eq. 19: Final token weight (confidence × alignment)
            β_t = (p_t * w_t) / sum(p_k * w_k for all k in generated_tokens)
    
    # Eq. 20: Joint token relevance (cross-modal reasoning indicator)
    FOR token_t in generated_tokens:
        γ_t = β_t_visual * β_t_prompt
    
    # === OPTIONAL: RELEVANCE FLOW REDISTRIBUTION ===
    
    IF redistribute_flow:
        # Eq. 21-22: Transfer relevance from function words to content words
        FOR function_word i in ['is', 'are', 'a', 'the', 'of', ...]:
            FOR content_word j after i:
                # Normalized influence from i to j
                F_ij = R[j,i] / sum(R[k,i] for k > i)
                
                # Actual relevance flow
                f_ij = β_i * F_ij
                
                # Update content word weight
                β_j_new = β_j + λ_flow * f_ij
        
        # L1 normalize updated weights
        β_new = L1_normalize(β_new)
    
    # === STAGE 4: HOLISTIC SALIENCY AGGREGATION ===
    
    # Eq. 23: Aggregate token-level relevance into unified maps
    
    # Visual saliency (highlights important image regions)
    R_tilde_V = sum(β_t_visual * R[t, V] for t in generated_tokens)
    
    # Prompt saliency (shows prompt token contributions)
    R_tilde_P = sum(β_t_prompt * R[t, P] for t in generated_tokens)
    
    # === RETURN EXPLANATIONS ===
    
    RETURN {
        'visual_saliency': R_tilde_V,         # Spatial heatmap over image patches
        'prompt_saliency': R_tilde_P,         # Token importance in prompt
        'cross_modal_relevance': γ_t,         # Cross-modal token scores
        'relevance_matrix': R,                # Full token-to-token relevance
        'token_weights': {                    # Individual token contributions
            'visual_weights': β_t_visual,
            'prompt_weights': β_t_prompt
        },
        'cross_modal_weights': cross_modal_weights,  # Alignment & confidence
        'layer_relevance': layer_relevance_matrices, # Per-layer attention
        'adaptive_weights': α_l               # Layer importance weights
    }
```

## Usage Instructions

### 1. Basic Usage

```python
# Initialize GLIMPSE explainer
from src.analysis.glimpse_explainer import GLIMPSEExplainer

explainer = GLIMPSEExplainer(
    model=your_vlm_model,
    tokenizer=your_tokenizer,
    lambda_head=1.0,      # Head weighting temperature
    lambda_depth=0.1,     # Depth prior temperature  
    lambda_flow=0.5       # Flow redistribution strength
)

# Run explanation
explanations = explainer.explain(
    input_ids=input_tokens,
    attention_mask=attention_mask,
    pixel_values=image_pixels,
    target_token_idx=-1,  # Last token
    visual_tokens=576,    # Number of image patches
    redistribute_flow=True
)

# Access results
visual_saliency = explanations['visual_saliency']      # Image heatmap
prompt_saliency = explanations['prompt_saliency']      # Text importance  
cross_modal_relevance = explanations['cross_modal_relevance']  # Token scores
```

### 2. Integration with Existing Framework

```python
# Run analysis using the integration script
python run_glimpse_analysis.py \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name image_text \
    --concept apple \
    --max_samples 10 \
    --output_dir results/glimpse \
    --glimpse_lambda_head 1.0 \
    --glimpse_lambda_depth 0.1 \
    --glimpse_redistribute_flow
```

### 3. Hook Integration

```python
# Add GLIMPSE to existing hook system
args.hook_names = ["glimpse_explanation"]

hook_return_functions, hook_postprocessing_functions = setup_hooks(
    model=model,
    modules_to_hook=[[".*attention.*"]],
    hook_names=args.hook_names,
    tokenizer=tokenizer,
    logger=logger,
    args=args
)
```

## Key Parameters

### GLIMPSE Parameters
- `λ_head` (lambda_head): Temperature for attention head weighting (default: 1.0)
- `λ_depth` (lambda_depth): Temperature for depth-based prior (default: 0.1)  
- `λ_flow` (lambda_flow): Flow strength for relevance redistribution (default: 0.5)
- `visual_tokens`: Number of visual tokens/patches (default: 576)
- `redistribute_flow`: Enable relevance flow from function words (default: False)

### Model Parameters  
- `model_name`: VLM model identifier
- `generation_mode`: Whether to use text generation (default: True)
- `max_new_tokens`: Maximum tokens to generate (default: 50)
- `target_token_idx`: Token index to explain (default: -1 for last token)

## Output Interpretation

### 1. Visual Saliency Map (`visual_saliency`)
- **Shape**: `[num_visual_patches]` 
- **Meaning**: Importance score for each image patch/region
- **Usage**: Overlay on original image as heatmap to show which regions the model focuses on

### 2. Prompt Saliency Map (`prompt_saliency`)  
- **Shape**: `[num_prompt_tokens]`
- **Meaning**: Importance score for each input prompt token
- **Usage**: Highlight which parts of the prompt drive the model's visual attention

### 3. Cross-Modal Token Relevance (`cross_modal_relevance`)
- **Shape**: `[num_generated_tokens]`
- **Meaning**: Joint alignment score combining visual and textual reasoning
- **Usage**: Identify tokens that exhibit strong multimodal reasoning

### 4. Relevance Matrix (`relevance_matrix`)
- **Shape**: `[sequence_length, sequence_length]`
- **Meaning**: Token-to-token relevance/interaction strengths
- **Usage**: Analyze fine-grained interactions between different modalities

## Advanced Features

### 1. Relevance Flow Redistribution
- Transfers relevance from function words ("is", "the", etc.) to content words
- Improves interpretability by emphasizing semantically meaningful tokens
- Controlled by `lambda_flow` parameter

### 2. Adaptive Layer Weighting
- Combines empirical gradient evidence with architectural depth priors
- Allows important layers to override depth bias when showing exceptional relevance
- Balances between data-driven and structure-based importance

### 3. Multi-Modal Alignment Analysis
- Separate analysis of prompt→visual and visual→prompt alignments
- Cross-modal token relevance identifies tokens with joint reasoning
- Confidence weighting emphasizes model's certain predictions

## Troubleshooting

### Common Issues

1. **Memory Issues**: Large models may require gradient checkpointing or reduced batch sizes
2. **Hook Registration**: Ensure attention modules are properly identified for your model architecture  
3. **Token Alignment**: Verify visual/prompt/generated token index boundaries are correct
4. **Gradient Flow**: Check that target token allows gradient backpropagation

### Performance Optimization

1. **Use Mixed Precision**: Enable fp16/bf16 for memory efficiency
2. **Limit Sequence Length**: Longer sequences increase computational complexity quadratically
3. **Batch Processing**: Process multiple samples sequentially rather than in parallel
4. **Cache Attention**: Reuse attention computations when possible

## File Structure

```
src/
├── analysis/
│   ├── glimpse_explainer.py      # Core GLIMPSE implementation
│   └── glimpse_integration.py    # Framework integration helpers
├── helpers/
│   └── utils.py                  # Updated with GLIMPSE hooks
└── models/                       # VLM model implementations
run_glimpse_analysis.py           # Complete usage example
```

This implementation provides a complete, production-ready GLIMPSE explainer that integrates seamlessly with the existing VLM analysis framework while maintaining the theoretical rigor of the original method.
