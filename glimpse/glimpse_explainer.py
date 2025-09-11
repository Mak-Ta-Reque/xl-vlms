"""
GLIMPSE: Gradient-weighted Layer-wise Importance for Multimodal Prompt-guided Saliency Explanation

This module implements the complete GLIMPSE method for explaining vision-language model decisions
through gradient-weighted attention analysis and cross-modal token relevance.

4-Stage GLIMPSE Algorithm:
0. Preliminaries: Encode input sequence x = [v1:K || p1:M || y1:T], forward/backward pass
1. Layer Relevance Extraction: Compute gradient-weighted attention G_l^h = ReLU(g_l^h) ⊙ A_l^h
2. Adaptive Layer Propagation: Weight layers by gradient magnitude and depth prior
3. Cross-Modal Token Relevancy: Compute alignment and confidence scores for generated tokens  
4. Holistic Saliency Aggregation: Generate visual and prompt saliency maps

Outputs:
- R̃V: Visual saliency map highlighting important image regions
- R̃P: Prompt saliency map showing text token contributions  
- γt: Cross-modal token relevance scores for generated tokens

Author: Implementation based on GLIMPSE methodology
"""

from pyexpat import model
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import math

class GLIMPSEExplainer:
    """
    GLIMPSE explainer for vision-language models.
    
    Provides gradient-weighted attention analysis to generate:
    1. Visual saliency maps highlighting important image regions
    2. Prompt saliency maps showing text token contributions
    3. Cross-modal token relevance scores for generated tokens
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Callable,
        lambda_head: float = 1.0,
        lambda_depth: float = 0.1,
        lambda_flow: float = 0.5,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        """
        Initialize GLIMPSE explainer.
        
        Args:
            model: The vision-language model to explain
            tokenizer: Tokenizer for the model
            lambda_head: Temperature parameter for head weighting (Eq. 6)
            lambda_depth: Temperature parameter for depth-based prior (Eq. 10) 
            lambda_flow: Flow strength parameter for relevance redistribution (Eq. 22)
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.lambda_head = lambda_head
        self.lambda_depth = lambda_depth
        self.lambda_flow = lambda_flow
        self.device = device
        
        # Storage for attention weights and gradients
        self.attention_weights = {}  # {layer: {head: attention_matrix}}
        self.attention_gradients = {}  # {layer: {head: gradient_matrix}}
        self.hooks = []
        self.patched_modules = {}  # Store original forward methods for restoration
        
        # Token indices for different modalities
        self.visual_indices = None  # V = {1, ..., K}
        self.prompt_indices = None  # P = {K+1, ..., K+M}  
        self.generated_indices = None  # Y = {K+M+1, ..., N}
        
    def register_hooks(self):
        """Register forward and backward hooks to capture attention weights and gradients."""
        self.hooks = []
        
        def attention_forward_hook(layer_idx, head_idx):
            def hook(module, input, output):
                # Store attention weights A^l_h - handle different model architectures
                attn_weights = None
                
                # Try different ways to extract attention weights
                if isinstance(output, tuple):
                    # Some models return (output, attention_weights, ...)
                    for item in output:
                        if isinstance(item, torch.Tensor) and item.dim() >= 3:
                            # Check if this looks like attention weights (batch, heads, seq, seq) or (batch, seq, seq)
                            if item.dim() == 4 or (item.dim() == 3 and item.shape[-1] == item.shape[-2]):
                                attn_weights = item
                                break
                elif hasattr(output, 'attn_weights') and output.attn_weights is not None:
                    # Standard attribute access
                    attn_weights = output.attn_weights
                elif hasattr(output, 'attention_weights') and output.attention_weights is not None:
                    # Alternative attribute name
                    attn_weights = output.attention_weights
                elif hasattr(module, 'attention_weights') and module.attention_weights is not None:
                    # Some models store on the module itself
                    attn_weights = module.attention_weights
                
                # Store if we found attention weights
                if attn_weights is not None:
                    if layer_idx not in self.attention_weights:
                        self.attention_weights[layer_idx] = {}
                    self.attention_weights[layer_idx][head_idx] = attn_weights.detach()
                    print(f"Captured attention weights for layer {layer_idx}, shape: {attn_weights.shape}")
                else:
                    print(f"No attention weights found for layer {layer_idx}, module: {type(module).__name__}")
            return hook
        
        def attention_backward_hook(layer_idx, head_idx):  
            def hook(module, grad_input, grad_output):
                # Store attention gradients ∂z_t/∂A^l_h
                if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
                    if layer_idx not in self.attention_gradients:
                        self.attention_gradients[layer_idx] = {}
                    self.attention_gradients[layer_idx][head_idx] = grad_output[0].detach()
                    print(f"Captured attention gradients for layer {layer_idx}, shape: {grad_output[0].shape}")
            return hook
        
        # Register hooks for all attention layers with more specific matching
        layer_idx = 0
        print("Registering hooks on attention modules...")
        
        for name, module in self.model.named_modules():
            # More specific patterns for attention modules
            attention_patterns = [
                'attention', 'attn', 'self_attn', 'cross_attn', 
                'multihead_attn', 'mha', 'self_attention'
            ]
            
            if any(pattern in name.lower() for pattern in attention_patterns):
                print(f"Found attention module: {name} ({type(module).__name__})")
                
                # Forward hook
                handle_fwd = module.register_forward_hook(
                    attention_forward_hook(layer_idx, 0)
                )
                self.hooks.append(handle_fwd)
                
                # Backward hook  
                handle_bwd = module.register_backward_hook(
                    attention_backward_hook(layer_idx, 0)
                )
                self.hooks.append(handle_bwd)
                
                layer_idx += 1
        
        print(f"Registered hooks on {layer_idx} attention modules")
        
        # If no attention modules were found, try alternative registration
        if layer_idx == 0:
            print("No attention modules found with standard patterns, trying alternative approach...")
            self._register_alternative_hooks()
    
    def _register_alternative_hooks(self):
        """Alternative hook registration for models that don't expose attention in standard ways."""
        layer_idx = 0
        
        # Try to find attention-like modules by their class names
        attention_classes = [
            'MultiheadAttention', 'Attention', 'SelfAttention', 'CrossAttention',
            'QwenAttention', 'LlamaAttention', 'GPTAttention', 'BertAttention'
        ]
        
        for name, module in self.model.named_modules():
            class_name = type(module).__name__
            if any(att_class in class_name for att_class in attention_classes):
                print(f"Found attention class: {name} ({class_name})")
                
                # Store original forward method for restoration
                original_forward = module.forward
                self.patched_modules[module] = original_forward
                
                def patched_forward(layer_idx=layer_idx, original=original_forward):
                    def forward_wrapper(*args, **kwargs):
                        # Call original forward
                        result = original(*args, **kwargs)
                        
                        # Try to extract attention weights from various sources
                        attn_weights = None
                        
                        if hasattr(module, 'attn_weights'):
                            attn_weights = module.attn_weights
                        elif hasattr(module, 'attention_weights'):
                            attn_weights = module.attention_weights  
                        elif isinstance(result, tuple) and len(result) > 1:
                            # Look for attention weights in return tuple
                            for item in result:
                                if isinstance(item, torch.Tensor) and item.dim() >= 3:
                                    if item.shape[-1] == item.shape[-2]:  # Square matrix indicates attention
                                        attn_weights = item
                                        break
                        
                        # Store attention weights if found
                        if attn_weights is not None:
                            if layer_idx not in self.attention_weights:
                                self.attention_weights[layer_idx] = {}
                            self.attention_weights[layer_idx][0] = attn_weights.detach()
                            print(f"Patched capture: attention weights for layer {layer_idx}, shape: {attn_weights.shape}")
                        
                        return result
                    return forward_wrapper
                
                # Apply the patch
                module.forward = patched_forward()
                layer_idx += 1
        
        print(f"Applied patches to {layer_idx} attention modules")
        
        # If still no attention captured, try SDPA-specific approach
        if layer_idx == 0:
            print("Trying SDPA-specific attention capture...")
            self._register_sdpa_hooks()
    
    def _register_sdpa_hooks(self):
        """Special hook registration for SDPA (Scaled Dot Product Attention) models."""
        layer_idx = 0
        
        # Look for specific SDPA patterns
        for name, module in self.model.named_modules():
            # Check for modules that might contain SDPA operations
            if any(pattern in name.lower() for pattern in ['attention', 'attn', 'self_attn']):
                print(f"Attempting SDPA hook on: {name} ({type(module).__name__})")
                
                # Hook into the module's computation graph
                def sdpa_hook(layer_idx=layer_idx):
                    def hook_fn(module, input, output):
                        # For SDPA models, attention weights might be in different places
                        # Try to capture from input tensors during computation
                        if isinstance(input, (tuple, list)) and len(input) >= 3:
                            # SDPA typically takes (query, key, value) as inputs
                            query, key, value = input[0], input[1], input[2]
                            
                            if query is not None and key is not None:
                                # Compute attention weights manually: softmax(Q @ K^T / sqrt(d))
                                with torch.no_grad():
                                    d_k = query.size(-1)
                                    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
                                    attn_weights = torch.softmax(scores, dim=-1)
                                    
                                    if layer_idx not in self.attention_weights:
                                        self.attention_weights[layer_idx] = {}
                                    self.attention_weights[layer_idx][0] = attn_weights.detach()
                                    print(f"SDPA capture: attention weights for layer {layer_idx}, shape: {attn_weights.shape}")
                    
                    return hook_fn
                
                # Register the SDPA hook
                handle = module.register_forward_hook(sdpa_hook())
                self.hooks.append(handle)
                layer_idx += 1
        
        print(f"Registered SDPA hooks on {layer_idx} modules")
    
    def _fallback_attention_capture(self, inputs, logits):
        """Fallback method to generate synthetic attention weights when capture fails."""
        print("Attempting fallback attention capture using gradient approximation...")
        
        try:
            # Use gradient-based approximation to create attention-like weights
            sequence_length = inputs['input_ids'].shape[1]
            
            # Compute gradients of output with respect to input embeddings
            if hasattr(self.model, 'get_input_embeddings'):
                embeddings = self.model.get_input_embeddings()
                input_embeds = embeddings(inputs['input_ids'])
                input_embeds.requires_grad_(True)
                
                # Re-run forward pass with gradient tracking on embeddings
                outputs = self.model(inputs_embeds=input_embeds, attention_mask=inputs.get('attention_mask'))
                target_output = outputs.logits[:, -1, :].sum()
                
                # Compute gradients
                grads = torch.autograd.grad(target_output, input_embeds, create_graph=False)[0]
                
                # Create synthetic attention weights from gradients
                # Normalize gradients to create attention-like patterns
                grad_norms = torch.norm(grads, dim=-1)  # (batch, seq_len)
                
                # Create attention matrix: each position attends to all positions based on gradient similarity
                synthetic_attention = torch.softmax(
                    grad_norms.unsqueeze(-1) * grad_norms.unsqueeze(-2), dim=-1
                )
                
                # Store synthetic attention weights
                self.attention_weights[0] = {0: synthetic_attention.detach()}
                print(f"Fallback: Created synthetic attention weights, shape: {synthetic_attention.shape}")
                
                return True
                
        except Exception as e:
            print(f"Fallback method failed: {e}")
            return False
        
        return False
    
    def _compute_gradients_with_autograd(self, logits: torch.Tensor, Y: List[int], gen_tokens: List[int]):
        """
        Compute attention gradients using torch.autograd.grad() for each generated token.
        This provides more precise control over gradient computation compared to backward().
        
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            Y: List of global indices for generated tokens
            gen_tokens: List of generated token IDs
        """
        print("Computing gradients using autograd.grad() for each attention layer...")
        
        # Collect all attention matrices that require gradients
        attention_matrices = []
        attention_layer_head_mapping = []
        
        for layer_idx, heads in self.attention_weights.items():
            for head_idx, attn_weights in heads.items():
                # Enable gradients on attention weights
                attn_weights.requires_grad_(True)
                attention_matrices.append(attn_weights)
                attention_layer_head_mapping.append((layer_idx, head_idx))
        
        if not attention_matrices:
            print("WARNING: No attention matrices found for gradient computation")
            return
        
        print(f"Computing gradients for {len(attention_matrices)} attention matrices")
        
        # Initialize gradient storage
        self.attention_gradients = {}
        
        # Compute gradients for each generated token
        for t_idx, global_t in enumerate(Y):
            if global_t >= logits.shape[1] or t_idx >= len(gen_tokens):
                continue
                
            print(f"Computing gradients for token {t_idx} at position {global_t}")
            
            # Target logit for this generated token
            if logits.dim() == 3:  # [batch, seq, vocab]
                target_logit = logits[0, global_t, gen_tokens[t_idx]]
            else:  # [seq, vocab] or other format
                target_logit = logits[global_t, gen_tokens[t_idx]]
            
            try:
                # Use autograd.grad() to compute gradients with respect to attention matrices
                gradients = torch.autograd.grad(
                    outputs=target_logit,
                    inputs=attention_matrices,
                    grad_outputs=None,
                    retain_graph=True,
                    create_graph=False,
                    only_inputs=True,
                    allow_unused=True
                )
                
                # Store gradients in the expected format
                for i, grad in enumerate(gradients):
                    if grad is not None:
                        layer_idx, head_idx = attention_layer_head_mapping[i]
                        
                        # Initialize gradient storage if needed
                        if layer_idx not in self.attention_gradients:
                            self.attention_gradients[layer_idx] = {}
                        
                        # Accumulate gradients across all generated tokens
                        if head_idx not in self.attention_gradients[layer_idx]:
                            self.attention_gradients[layer_idx][head_idx] = grad.detach().clone()
                        else:
                            # Sum gradients from all generated tokens
                            self.attention_gradients[layer_idx][head_idx] += grad.detach()
                        
                        print(f"  Computed gradient for layer {layer_idx}, head {head_idx}, shape: {grad.shape}")
                    else:
                        layer_idx, head_idx = attention_layer_head_mapping[i]
                        print(f"  No gradient for layer {layer_idx}, head {head_idx} (unused)")
                        
            except Exception as e:
                print(f"Error computing gradients for token {t_idx}: {e}")
                continue
        
        # Normalize gradients by number of generated tokens for fair comparison
        num_tokens = len([t for t in Y if t < logits.shape[1]])
        if num_tokens > 0:
            for layer_idx in self.attention_gradients:
                for head_idx in self.attention_gradients[layer_idx]:
                    self.attention_gradients[layer_idx][head_idx] /= num_tokens
        
        print(f"Gradient computation complete. Captured gradients for {len(self.attention_gradients)} layers")

    def _compute_gradients_with_autograd_per_token(
        self, 
        logits: torch.Tensor, 
        Y: List[int], 
        gen_tokens: List[int],
        target_token_idx: int = -1
    ):
        """
        Alternative method to compute gradients for a specific target token using autograd.grad().
        
        Args:
            logits: Model output logits
            Y: List of global indices for generated tokens  
            gen_tokens: List of generated token IDs
            target_token_idx: Index of the target token to compute gradients for (-1 for last token)
        """
        print(f"Computing gradients for target token {target_token_idx} using autograd.grad()...")
        
        # Select target token
        if target_token_idx == -1:
            target_token_idx = len(Y) - 1
        
        if target_token_idx >= len(Y) or target_token_idx < 0:
            print(f"Invalid target token index: {target_token_idx}")
            return
            
        global_t = Y[target_token_idx]
        token_id = gen_tokens[target_token_idx]
        
        if global_t >= logits.shape[1]:
            print(f"Target token position {global_t} exceeds sequence length {logits.shape[1]}")
            return
        
        # Target logit
        target_logit = logits[0, global_t, token_id]
        print(f"Target logit value: {target_logit.item():.4f}")
        
        # Collect attention matrices with gradients enabled
        attention_matrices = []
        attention_layer_head_mapping = []
        
        for layer_idx, heads in self.attention_weights.items():
            for head_idx, attn_weights in heads.items():
                attn_weights.requires_grad_(True)
                attention_matrices.append(attn_weights)
                attention_layer_head_mapping.append((layer_idx, head_idx))
        
        if not attention_matrices:
            print("No attention matrices available for gradient computation")
            return
        
        try:
            # Compute gradients using autograd.grad()
            gradients = torch.autograd.grad(
                outputs=target_logit,
                inputs=attention_matrices,
                grad_outputs=None,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
                allow_unused=True
            )
            
            # Store gradients
            self.attention_gradients = {}
            for i, grad in enumerate(gradients):
                if grad is not None:
                    layer_idx, head_idx = attention_layer_head_mapping[i]
                    
                    if layer_idx not in self.attention_gradients:
                        self.attention_gradients[layer_idx] = {}
                    
                    self.attention_gradients[layer_idx][head_idx] = grad.detach()
                    print(f"Gradient computed for layer {layer_idx}, head {head_idx}, shape: {grad.shape}")
            
            print(f"Successfully computed gradients for {len(self.attention_gradients)} layers")
            
        except Exception as e:
            print(f"Error in autograd.grad() computation: {e}")
            import traceback
            traceback.print_exc()

    def _compute_autograd_attention_gradients(self, target_output: torch.Tensor):
        """
        Legacy method - kept for backward compatibility.
        Use _compute_gradients_with_autograd instead for better control.
        """
        print("Using legacy autograd gradient computation...")
        # This method is now deprecated in favor of _compute_gradients_with_autograd
        pass

    def clear_hooks(self):
        """Remove all registered hooks and restore patched modules."""
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Restore patched modules
        for module, original_forward in self.patched_modules.items():
            module.forward = original_forward
        self.patched_modules = {}
        
        # Clear stored data
        self.attention_weights = {}
        self.attention_gradients = {}
    
    def debug_attention_capture(self):
        """Debug method to check if attention weights are being captured."""
        print(f"Attention weights captured for {len(self.attention_weights)} layers")
        for layer_idx, heads in self.attention_weights.items():
            for head_idx, weights in heads.items():
                print(f"  Layer {layer_idx}, Head {head_idx}: {weights.shape}")
        
        print(f"Attention gradients captured for {len(self.attention_gradients)} layers")
        for layer_idx, heads in self.attention_gradients.items():
            for head_idx, grads in heads.items():
                print(f"  Layer {layer_idx}, Head {head_idx}: {grads.shape}")
                
        if len(self.attention_weights) == 0:
            print("WARNING: No attention weights captured! Check model architecture and hook registration.")
        
        return len(self.attention_weights) > 0
    
    def check_model_compatibility(self):
        """Check model compatibility and provide recommendations."""
        print("=== Model Compatibility Check ===")
        
        if hasattr(self.model, 'config'):
            config = self.model.config
            print(f"Model type: {type(self.model).__name__}")
            print(f"Config type: {type(config).__name__}")
            
            # Check attention implementation
            if hasattr(config, 'attn_implementation'):
                print(f"Attention implementation: {config.attn_implementation}")
                if config.attn_implementation == 'sdpa':
                    print("⚠️  SDPA implementation detected - may need special handling")
            
            # Check if attention output is supported
            if hasattr(config, 'output_attentions'):
                print(f"Output attentions supported: {hasattr(config, 'output_attentions')}")
            
        # List attention modules found
        attention_modules = []
        for name, module in self.model.named_modules():
            if any(pattern in name.lower() for pattern in ['attention', 'attn', 'self_attn']):
                attention_modules.append((name, type(module).__name__))
        
        print(f"Found {len(attention_modules)} potential attention modules:")
        for name, class_name in attention_modules[:5]:  # Show first 5
            print(f"  - {name}: {class_name}")
        if len(attention_modules) > 5:
            print(f"  ... and {len(attention_modules) - 5} more")
        
        print("=================================")
    
    def force_attention_output(self):
        """Force models to output attention weights by modifying their configuration."""
        # Try to enable attention output in the model configuration
        if hasattr(self.model, 'config'):
            config = self.model.config
            
            # Check if model uses SDPA attention implementation
            if hasattr(config, 'attn_implementation') and config.attn_implementation == 'sdpa':
                print(f"Model uses SDPA attention implementation. Switching to 'eager' for attention output...")
                try:
                    config.attn_implementation = 'eager'
                    print("Successfully switched to 'eager' attention implementation")
                except Exception as e:
                    print(f"Failed to switch attention implementation: {e}")
                    print("Will proceed with hook-based capture method")
            
            # Common attention output flags
            attention_flags = [
                'output_attentions', 'return_attention', 'output_attention_weights',
                'use_attention_mask', 'output_hidden_states'
            ]
            
            for flag in attention_flags:
                if hasattr(config, flag):
                    try:
                        original_value = getattr(config, flag)
                        setattr(config, flag, True)
                        print(f"Set {flag} = True (was {original_value})")
                    except ValueError as e:
                        if 'attn_implementation' in str(e) and 'sdpa' in str(e):
                            print(f"Cannot set {flag} with SDPA implementation: {e}")
                            # Try to switch to eager implementation first
                            if hasattr(config, 'attn_implementation'):
                                try:
                                    config.attn_implementation = 'eager'
                                    setattr(config, flag, True)
                                    print(f"Switched to eager implementation and set {flag} = True")
                                except Exception as e2:
                                    print(f"Failed to switch implementation and set {flag}: {e2}")
                        else:
                            print(f"Error setting {flag}: {e}")
                    except Exception as e:
                        print(f"Unexpected error setting {flag}: {e}")
        
        # Also try to set it directly on the model
        if hasattr(self.model, 'output_attentions'):
            try:
                self.model.output_attentions = True
                print("Set model.output_attentions = True")
            except Exception as e:
                print(f"Error setting model.output_attentions: {e}")
    
    def set_token_indices(
        self, 
        visual_tokens: int, 
        prompt_tokens: int, 
        generated_tokens: int
    ):
        """
        Set token indices for different modalities.
        
        Args:
            visual_tokens: Number of visual tokens (K)
            prompt_tokens: Number of prompt tokens (M) 
            generated_tokens: Number of generated tokens (T)
        """
        K, M, T = visual_tokens, prompt_tokens, generated_tokens
        
        # Eq. 2-4: Index sets for different modalities
        self.visual_indices = list(range(0, K))  # V = {0, ..., K-1}
        self.prompt_indices = list(range(K, K + M))  # P = {K, ..., K+M-1}
        self.generated_indices = list(range(K + M, K + M + T))  # Y = {K+M, ..., K+M+T-1}
    
    def compute_layer_relevance(self, layer_idx: int) -> torch.Tensor:
        """
        Stage 1: Layer Relevance Extraction
        Compute fused attention matrix for a layer using gradient-weighted attention.
        
        Args:
            layer_idx: Index of the layer
            
        Returns:
            E^l: Fused attention matrix for layer l (Eq. 8)
        """
        if layer_idx not in self.attention_weights or layer_idx not in self.attention_gradients:
            raise ValueError(f"No attention data found for layer {layer_idx}")
        
        attention_layer = self.attention_weights[layer_idx]
        gradient_layer = self.attention_gradients[layer_idx]
        
        # Compute head weights for this layer
        head_weights = []
        gradient_weighted_attentions = []
        
        for head_idx in attention_layer.keys():
            if head_idx not in gradient_layer:
                continue
                
            A_lh = attention_layer[head_idx]  # Attention matrix
            g_lh = gradient_layer[head_idx]   # Gradient matrix
            
            # Eq. 5: Element-wise product with positive gradients
            G_lh = F.relu(g_lh) * A_lh
            gradient_weighted_attentions.append(G_lh)
            
            # Eq. 6-7: Compute head importance weight
            numerator = G_lh.sum()  # Sum of G^l_h(i,j)
            denominator = F.relu(g_lh).sum()  # Sum of ReLU(g^l_h(i,j))
            
            if denominator > 0:
                head_importance = numerator / denominator
            else:
                head_importance = torch.tensor(0.0)
                
            head_weights.append(head_importance)
        
        if not head_weights:
            raise ValueError(f"No valid attention heads found for layer {layer_idx}")
        
        # Eq. 6: Softmax normalization of head weights
        head_weights = torch.stack(head_weights)
        head_weights = F.softmax(head_weights / self.lambda_head, dim=0)
        
        # Eq. 8: Weighted combination of gradient-weighted attentions  
        E_l = torch.zeros_like(gradient_weighted_attentions[0])
        for i, G_lh in enumerate(gradient_weighted_attentions):
            E_l += head_weights[i] * G_lh
        
        # Row normalize to preserve probability mass
        row_sums = E_l.sum(dim=-1, keepdim=True)
        row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
        E_l = E_l / row_sums
        
        return E_l

    def compute_clip_style_relevance_matrix(
        self, 
        start_layer: int = 6,
        start_layer_text: int = 6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute relevance matrices using CLIP-style approach with autograd gradients.
        This method computes token-to-token relevance similar to the CLIP interpretation method.
        
        Args:
            start_layer: Starting layer for image attention analysis
            start_layer_text: Starting layer for text attention analysis
            
        Returns:
            Tuple of (image_relevance, text_relevance) tensors
        """
        print(f"Computing CLIP-style relevance starting from layer {start_layer}")
        
        # Find available attention layers
        image_layers = []
        text_layers = []
        
        for layer_idx in sorted(self.attention_weights.keys()):
            if layer_idx >= start_layer:
                # For multimodal models, we'll treat all layers as potentially containing both modalities
                if layer_idx in self.attention_weights and layer_idx in self.attention_gradients:
                    image_layers.append(layer_idx)
                    text_layers.append(layer_idx)
        
        print(f"Found {len(image_layers)} layers for relevance computation")
        
        if not image_layers:
            print("No layers available for relevance computation")
            return None, None
            
        # Get sequence length from first available attention matrix
        first_layer = image_layers[0]
        first_head = list(self.attention_weights[first_layer].keys())[0]
        attention_shape = self.attention_weights[first_layer][first_head].shape
        
        batch_size = attention_shape[0] if len(attention_shape) == 4 else 1
        num_tokens = attention_shape[-1]
        
        print(f"Sequence length: {num_tokens}, Batch size: {batch_size}")
        
        # Initialize relevance matrix as identity (similar to CLIP approach)
        R = torch.eye(num_tokens, num_tokens, dtype=torch.float32, device=self.device)
        R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        
        # Process each attention layer
        for layer_idx in image_layers:
            if layer_idx not in self.attention_weights or layer_idx not in self.attention_gradients:
                continue
                
            print(f"Processing layer {layer_idx}")
            
            # Get attention weights and gradients for this layer
            layer_attention = self.attention_weights[layer_idx]
            layer_gradients = self.attention_gradients[layer_idx] 
            
            # Aggregate across heads (similar to CLIP method)
            layer_cam = None
            
            for head_idx in layer_attention.keys():
                if head_idx not in layer_gradients:
                    continue
                    
                attn_weights = layer_attention[head_idx]
                grad = layer_gradients[head_idx]
                
                # Ensure proper shape handling
                if len(attn_weights.shape) == 4:  # (batch, heads, seq, seq)
                    attn_weights = attn_weights.reshape(-1, attn_weights.shape[-2], attn_weights.shape[-1])
                    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
                elif len(attn_weights.shape) == 3:  # (batch, seq, seq) or (heads, seq, seq)
                    if attn_weights.shape[0] != batch_size:
                        # Likely (heads, seq, seq), expand for batch
                        attn_weights = attn_weights.unsqueeze(0).expand(batch_size, -1, -1, -1)
                        attn_weights = attn_weights.reshape(-1, attn_weights.shape[-2], attn_weights.shape[-1])
                        grad = grad.unsqueeze(0).expand(batch_size, -1, -1, -1)  
                        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
                
                # Compute gradient-weighted attention (CLIP-style)
                cam = grad.detach() * attn_weights.detach()
                cam = cam.clamp(min=0)  # ReLU activation
                
                # Reshape back to batch format and average over heads
                cam = cam.reshape(batch_size, -1, cam.shape[-2], cam.shape[-1])
                cam = cam.mean(dim=1)  # Average over heads
                
                if layer_cam is None:
                    layer_cam = cam
                else:
                    layer_cam += cam
            
            if layer_cam is not None:
                # Update relevance matrix using matrix multiplication (CLIP approach)
                R = R + torch.bmm(layer_cam, R)
                print(f"Updated relevance matrix with layer {layer_idx}, shape: {R.shape}")
        
        # Extract image and text relevance (assuming tokens are ordered: [CLS, image_tokens..., text_tokens...])
        if self.visual_indices and self.prompt_indices:
            # Image relevance: from CLS token to visual tokens
            image_relevance = R[:, 0, self.visual_indices]  # (batch, num_visual_tokens)
            
            # Text relevance: full text-to-text relevance matrix
            text_start = min(self.prompt_indices) if self.prompt_indices else 1
            text_end = max(self.prompt_indices) + 1 if self.prompt_indices else num_tokens
            text_relevance = R[:, text_start:text_end, text_start:text_end]  # (batch, text_len, text_len)
        else:
            # Fallback: assume first half is visual, second half is text
            mid_point = num_tokens // 2
            image_relevance = R[:, 0, 1:mid_point]  # From CLS to visual tokens
            text_relevance = R[:, mid_point:, mid_point:]  # Text-to-text relevance
        
        print(f"Image relevance shape: {image_relevance.shape}")
        print(f"Text relevance shape: {text_relevance.shape}")
        
        return image_relevance, text_relevance
    
    def compute_adaptive_layer_weights(self, num_layers: int) -> torch.Tensor:
        """
        Stage 2: Adaptive Layer Propagation - Compute layer weights
        Combines gradient magnitude and depth-based prior.
        
        Args:
            num_layers: Total number of layers (L)
            
        Returns:
            alpha: Layer weights α^l (Eq. 11)
        """
        # Handle edge case
        if num_layers == 0:
            return torch.tensor([], device=self.device)
        gradient_norms = []
        
        # Eq. 9: Compute gradient norm for each layer
        for layer_idx in range(num_layers):
            if layer_idx in self.attention_gradients:
                layer_gradients = []
                for head_idx, g_lh in self.attention_gradients[layer_idx].items():
                    layer_gradients.append(g_lh)
                
                if layer_gradients:
                    # L1 norm of aggregated gradients
                    aggregated_grad = torch.stack(layer_gradients).sum(dim=0)
                    g_l = torch.norm(aggregated_grad, p=1)
                    gradient_norms.append(g_l)
                else:
                    gradient_norms.append(torch.tensor(0.0, device=self.device))
            else:
                gradient_norms.append(torch.tensor(0.0, device=self.device))
        
        # Ensure all tensors are on the same device
        gradient_norms = torch.stack([g.to(self.device) for g in gradient_norms])
        
        # Eq. 10: Depth-based prior (higher weight for deeper layers)
        depth_weights = []
        for layer_idx in range(num_layers):
            depth_value = self.lambda_depth * (layer_idx + 1)
            s_l = torch.exp(torch.tensor(depth_value, dtype=torch.float32, device=self.device))
            depth_weights.append(s_l)
        
        depth_weights = torch.stack(depth_weights)
        depth_weights = depth_weights / depth_weights.sum()
        
        # Eq. 11: Combined adaptive weights
        combined_weights = gradient_norms * depth_weights
        alpha = combined_weights / (combined_weights.sum() + 1e-8)  # Add epsilon to avoid division by zero
        
        return alpha
    
    def propagate_relevance(self, num_layers: int, sequence_length: int) -> torch.Tensor:
        """
        Stage 2: Adaptive Layer Propagation - Propagate relevance through layers
        
        Args:
            num_layers: Total number of layers (L)
            sequence_length: Length of input sequence (N)
            
        Returns:
            R: Final relevance matrix (N × N)
        """
        # Eq. 12: Initialize relevance matrix as identity
        R = torch.eye(sequence_length, device=self.device)
        
        # Handle edge case where no layers are available
        if num_layers == 0:
            print("Warning: No attention layers found, returning identity matrix")
            return R
        
        # Get adaptive layer weights
        alpha = self.compute_adaptive_layer_weights(num_layers)
        
        # Check if we have any non-zero alpha values
        if torch.all(alpha == 0):
            print("Warning: All layer weights are zero, returning identity matrix")
            return R
        
        successful_layers = 0
        # Sequential propagation through layers
        for layer_idx in range(num_layers):
            try:
                # Get layer relevance matrix
                E_l = self.compute_layer_relevance(layer_idx)
                
                # Ensure E_l has the correct shape
                if E_l.shape[0] != sequence_length or E_l.shape[1] != sequence_length:
                    print(f"Warning: Layer {layer_idx} attention matrix shape {E_l.shape} doesn't match sequence length {sequence_length}")
                    continue
                
                # Eq. 13: Layer-specific transformation
                L_l = torch.eye(sequence_length, device=self.device) + alpha[layer_idx] * E_l.to(self.device)
                
                # Eq. 14: Additive accumulation  
                R = R + L_l @ R
                successful_layers += 1
                
            except (ValueError, KeyError) as e:
                # Skip layers without attention data
                print(f"Skipping layer {layer_idx}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error in layer {layer_idx}: {e}")
                continue
        
        print(f"Successfully processed {successful_layers}/{num_layers} attention layers")
        return R
    
    def compute_cross_modal_weights(
        self, 
        relevance_matrix: torch.Tensor,
        logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 3: Cross-Modal Token Relevancy
        Compute alignment and confidence weights for generated tokens.
        
        Args:
            relevance_matrix: Propagated relevance matrix R (N × N) 
            logits: Model logits for generated tokens
            
        Returns:
            Dictionary containing alignment weights and confidence scores
        """
        if self.generated_indices is None:
            raise ValueError("Generated token indices not set")
        
        results = {}
        
        # Extract relevance for generated tokens
        generated_relevance = relevance_matrix[self.generated_indices, :]
        
        # Eq. 15: Prompt alignment weights  
        if self.prompt_indices:
            prompt_relevance = generated_relevance[:, self.prompt_indices]
            a_t = prompt_relevance.mean(dim=1)  # Average over prompt tokens
            results['prompt_alignment'] = a_t
        
        # Eq. 16: Visual alignment weights
        if self.visual_indices:
            visual_relevance = generated_relevance[:, self.visual_indices]  
            v_t = visual_relevance.mean(dim=1)  # Average over visual tokens
            results['visual_alignment'] = v_t
        
        # Eq. 17: Confidence weights (softmax probabilities)
        p_t = F.softmax(logits, dim=-1)
        results['confidence'] = p_t
        
        return results
    
    def compute_token_relevance_scores(
        self, 
        cross_modal_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute final token relevance scores and saliency maps.
        
        Args:
            cross_modal_weights: Output from compute_cross_modal_weights
            
        Returns:
            Dictionary containing token relevance scores and saliency maps
        """
        results = {}
        
        # Extract weights
        a_t = cross_modal_weights.get('prompt_alignment')
        v_t = cross_modal_weights.get('visual_alignment') 
        p_t = cross_modal_weights.get('confidence')
        
        if a_t is not None and p_t is not None:
            # Eq. 19: Token weights for visual saliency (m = V)
            beta_t_V = (p_t * a_t) / (p_t * a_t).sum()
            results['visual_saliency_weights'] = beta_t_V
        
        if v_t is not None and p_t is not None:
            # Eq. 19: Token weights for prompt saliency (m = P) 
            beta_t_P = (p_t * v_t) / (p_t * v_t).sum()
            results['prompt_saliency_weights'] = beta_t_P
        
        if a_t is not None and v_t is not None:
            # Eq. 20: Joint token relevance
            gamma_t = beta_t_V * beta_t_P if 'visual_saliency_weights' in results else None
            if gamma_t is not None:
                results['cross_modal_relevance'] = gamma_t
        
        return results
    
    def redistribute_relevance_flow(
        self, 
        relevance_matrix: torch.Tensor,
        token_weights: torch.Tensor,
        function_words: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Optional relevance redistribution from function words to content words.
        
        Args:
            relevance_matrix: Current relevance matrix R
            token_weights: Current token weights β_t  
            function_words: List of function words to redistribute from
            
        Returns:
            Updated token weights β'_t (Eq. 22)
        """
        if function_words is None:
            function_words = ['is', 'are', 'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to']
        
        updated_weights = token_weights.clone()
        seq_length = relevance_matrix.shape[0]
        
        for i in range(seq_length):
            for j in range(i + 1, seq_length):
                # Eq. 21: Compute flow from token i to token j
                flow_denominator = relevance_matrix[i+1:, i].sum()
                if flow_denominator > 0:
                    F_ij = relevance_matrix[j, i] / flow_denominator  # Normalized influence
                    f_ij = token_weights[i] * F_ij  # Actual flow
                    
                    # Eq. 22: Update weights with received flow
                    updated_weights[j] += self.lambda_flow * f_ij
        
        # L1 normalization
        updated_weights = updated_weights / updated_weights.sum()
        
        return updated_weights
    
    def _redistribute_flow(
        self, 
        R_tilde_P: torch.Tensor, 
        R: torch.Tensor, 
        P: List[int], 
        function_words: List[str], 
        lambda_flow: float
    ) -> torch.Tensor:
        """
        Helper method for relevance flow redistribution from function words to content words.
        
        Args:
            R_tilde_P: Current prompt saliency map
            R: Full relevance matrix
            P: Prompt token indices
            function_words: List of function words to redistribute from
            lambda_flow: Flow strength parameter
            
        Returns:
            Updated prompt saliency map with redistributed relevance
        """
        updated_saliency = R_tilde_P.clone()
        
        # Get function word indices in the prompt
        prompt_tokens = [self.tokenizer.decode([i]) for i in P]
        function_indices = []
        
        for i, token in enumerate(prompt_tokens):
            if token.lower().strip() in function_words:
                function_indices.append(i)
        
        # Redistribute relevance from function words to content words
        for func_idx in function_indices:
            func_relevance = updated_saliency[func_idx]
            
            # Find content words after this function word
            for content_idx in range(func_idx + 1, len(P)):
                if content_idx not in function_indices:
                    # Compute flow based on relevance matrix
                    if func_idx < R.shape[0] and content_idx < R.shape[1]:
                        flow_strength = R[P[content_idx], P[func_idx]]
                        flow_amount = lambda_flow * func_relevance * flow_strength
                        
                        # Transfer relevance
                        updated_saliency[content_idx] += flow_amount
                        updated_saliency[func_idx] -= flow_amount * 0.5  # Partial reduction
        
        # Ensure non-negative values
        updated_saliency = torch.clamp(updated_saliency, min=0.0)
        
        # L1 normalize
        updated_saliency = updated_saliency / (updated_saliency.sum() + 1e-8)
        
        return updated_saliency
    
    def compute_holistic_saliency(
        self,
        relevance_matrix: torch.Tensor, 
        token_weights: torch.Tensor,
        modality: str = 'visual'
    ) -> torch.Tensor:
        """
        Stage 3: Holistic Saliency Aggregation
        Aggregate individual token relevance maps into unified saliency.
        
        Args:
            relevance_matrix: Final relevance matrix R (N × N)
            token_weights: Token weights β_t^(m) (Eq. 19)
            modality: Target modality ('visual' or 'prompt')
            
        Returns:
            R_tilde_m: Holistic saliency map (Eq. 23)
        """
        if modality == 'visual' and self.visual_indices:
            target_indices = self.visual_indices
        elif modality == 'prompt' and self.prompt_indices:
            target_indices = self.prompt_indices
        else:
            raise ValueError(f"Invalid modality '{modality}' or indices not set")
        
        # Eq. 23: Weighted aggregation over generated tokens
        holistic_saliency = torch.zeros(len(target_indices), device=self.device)
        
        for t_idx, weight in enumerate(token_weights):
            if t_idx < len(self.generated_indices):
                global_t_idx = self.generated_indices[t_idx]
                # R(t, m) - relevance from token t to target modality m
                token_relevance = relevance_matrix[global_t_idx, target_indices]
                holistic_saliency += weight * token_relevance
        
        return holistic_saliency
    


    def explain(
        self,
        inputs: Dict[str, torch.Tensor],
        generated_tokens: List[str],
        textual_prompt: str = "",
        attention_mask: Optional[torch.Tensor] = None, 
        pixel_values: Optional[torch.Tensor] = None,
        visual_tokens: int = 576,  # Common for many VLMs
        redistribute_flow: bool = False,
        target_token_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Main explain method that uses autograd.grad() for gradient computation.
        
        Args:
            inputs: Model inputs dictionary containing input_ids, attention_mask, etc.
            generated_tokens: List of generated token strings
            textual_prompt: The original textual prompt
            attention_mask: Attention mask (optional, can be in inputs)
            pixel_values: Image pixel values (optional, can be in inputs) 
            visual_tokens: Number of visual tokens
            redistribute_flow: Whether to apply relevance flow redistribution
            target_token_idx: Target token index for gradient computation
            
        Returns:
            Dictionary containing GLIMPSE explanation results
        """
        print("=== GLIMPSE Explanation with autograd.grad() ===")
        
        # Use the new autograd-based method
        return self.glimpse_explain(
            input_image=pixel_values,
            textual_prompt=textual_prompt,
            generated_tokens=generated_tokens,
            redistribute_flow=redistribute_flow
        )

    def create_simplified_explain_method(
        self,
        inputs: Dict[str, torch.Tensor],
        target_text: str = "",
        generated_response: str = "",
        image_token_count: int = 576
    ) -> Dict[str, Any]:
        """
        Simplified GLIMPSE explanation method for integration with existing systems.
        
        Args:
            inputs: Model inputs dictionary containing input_ids, attention_mask, etc.
            target_text: The textual prompt/question
            generated_response: The model's generated response
            image_token_count: Number of visual tokens (default 576 for 24x24 patches)
            
        Returns:
            Dictionary containing GLIMPSE explanation results
        """
        # Extract tokens from generated response
        if generated_response:
            generated_tokens = generated_response.split()
        else:
            generated_tokens = ["response"]  # Default single token
        
        # Call main explain method
        return self.explain(
            inputs=inputs,
            generated_tokens=generated_tokens,
            textual_prompt=target_text,
            redistribute_flow=False
        )
    
    start_layer =  -1
    start_layer_text =  -1



    def l1_norm(self, tensor):
        """
        Apply L1 normalization to a tensor.
        
        Args:
            tensor: Input tensor of shape [B, N, N]
            
        Returns:
            L1 normalized tensor where each matrix sums to 1
        """
        # Compute L1 norm (sum of absolute values) for each batch
        l1_norms = torch.sum(torch.abs(tensor), dim=(-2, -1), keepdim=True)  # [B, 1, 1]
        
        # Avoid division by zero
        l1_norms = torch.clamp(l1_norms, min=1e-8)
        
        # Normalize
        normalized_tensor = tensor / l1_norms
        
        return normalized_tensor

    def compute_local_relevance(self, G, A):
        """
        G, A: dictionaries with keys = layer names
            values = tensors of shape [B, H, N, N]
        Returns:
            E: dictionary with fused and row-normalized local relevance per layer
            shape [B, N, N]
        """
        E = {}
        g = []
        for layer_name in G.keys():
            g_l = G[layer_name]  # [B, H, N, N]
            a_l = A[layer_name]  # [B, H, N, N]

            # Local relevance: apply ReLU on the element-wise product
            G_l_h = F.relu(g_l * a_l)  # [B, H, N, N]

            # Head importance scores
            numerator = G_l_h.sum(dim=(-2))       # [B, H]
            
            denominator = F.relu(g_l).sum(dim=(-2)) + 1e-8 
            head_scores = numerator / denominator     # [B, H]
            # show head score as na imaeg of shape [B, H]

            #  

            print(f"Head scores for layer {layer_name}: {head_scores}")
            # Normalize head scores across heads
            w_l_h = F.softmax(head_scores, dim=1)    # [B, H]

            # Fuse heads
           
            E_l = ( G_l_h * w_l_h ).sum(dim=-1)           # [B, N, N]

            # Row-normalize: each row sums to 1
            #E_l = E_l / (E_l.sum(dim=-1, keepdim=True) + 1e-8)

            E[layer_name] = E_l  # [B, N, N]
            #aggregated_attention_g_l = g_l.sum(dim=-1)  # [B, N, N]

            #g.append(aggregated_attention_g_l)
        
       
        #print the g_dict keys and shapes
    
        return E
    
    def aggregated_attention_gradient(self, G):
        grads = [grad.sum(dim=-1) for grad in G.values() ]
        grads_keys = list(G.keys())
        g = torch.stack(grads, dim=0)  # [L, B, N, N]
        norm = g.abs().sum(dim=0, keepdim=True) + 1e-12
        g = g / norm
        # convert  Stack to list of g and then create the dict of layer name and corresponding g
        g_dict = {grads_keys[i]: g[i] for i in range(len(grads_keys))}
        return g_dict
    

    def compute_depth_weights(self, num_layers, lambda_d=1.0):
        """
        Compute depth-based layer weights using exponential weighting.
        
        Args:
            num_layers (int): Total number of layers L
            lambda_d (float): Depth scaling factor
            
        Returns:
            torch.Tensor: Layer weights s_l of shape [L] where s_l sums to 1
        """
        # Create layer indices [1, 2, ..., L]
        layer_indices = torch.arange(1, num_layers + 1, dtype=torch.float32)
        
        # Compute exponentials: exp(λd * (l+1)) for each layer l
        # Note: layer_indices already contains (l+1) since we start from 1
        exponentials = torch.exp(lambda_d * layer_indices)
        
        # Compute denominator: Σ_k exp(λd * (k+1))
        denominator = torch.sum(exponentials)
        
        # Compute normalized weights: s_l = exp(λd*(l+1)) / Σ_k exp(λd*(k+1))
        s_l = exponentials / denominator
    
        return s_l
    
    def compute_adaptive_layer_weights(self, g_dict, s_list):
        """
        Compute adaptive layer weights α_l = (g_l * s_l) / Σ_k (g_k * s_k)
        
        Args:
            g_dict (dict): Dictionary with layer names as keys and gradient magnitudes as values
                        e.g., {'layer_0': tensor_value, 'layer_1': tensor_value, ...}
            s_list (list or torch.Tensor): Depth weights [s_0, s_1, s_2, ...]
            
        Returns:
            dict: Dictionary with same keys as g_dict, values are α_l weights
        """
        layer_names = list(g_dict.keys())
        num_layers = len(layer_names)
        
        # Ensure s_list has correct length
        if len(s_list) != num_layers:
            raise ValueError(f"Length mismatch: g_dict has {num_layers} layers, s_list has {len(s_list)} elements")
        
        # Convert s_list to tensor if it's not already
        if not isinstance(s_list, torch.Tensor):
            s_tensor = torch.tensor(s_list, dtype=torch.float32)
        else:
            s_tensor = s_list
        
        # Compute g_l * s_l for each layer
        g_tensor = torch.stack(list(g_dict.values()), dim=1)
        B, L, I = g_tensor.shape
        s_tensor_expanded = s_tensor.view(1, L, 1).repeat(B, 1, I)
        numerators = g_tensor * s_tensor_expanded.to(g_tensor.device)
        
        # Convert to tensor for easier computation
   
        
        # Compute denominator: Σ_k (g_k * s_k)
        denominator = torch.sum(numerators, dim=1)  # Scalar
        
        # Compute α_l = (g_l * s_l) / Σ_k (g_k * s_k)
        alpha_tensor = numerators / (denominator + 1e-8)  # Add epsilon for numerical stability
        alpha_tensor = alpha_tensor.permute(1, 0, 2)
        # Create output dictionary with same keys as input
        alpha_dict = {}
        for i, layer_name in enumerate(layer_names):
            alpha_dict[layer_name] = alpha_tensor[i]
        
        return alpha_dict


    def tensor_to_image(self, tensor, filename='tensor_image.png', cmap='viridis'):
        """
        Visualize a 1D tensor (shape [1, N] or [N]) as a 2D image and save it.

        Args:
            tensor (torch.Tensor): Input tensor of shape [1, N] or [N].
            filename (str): File path to save the image.
            cmap (str): Colormap for visualization.
        """
        # Remove batch dimension if present
        if tensor.dim() == 2 and tensor.shape[0] == 1:
            tensor = tensor[0]
        
        N = tensor.numel()
        
        # Compute roughly square shape
        width = math.ceil(math.sqrt(N))
        height = math.ceil(N / width)
        
        # Pad tensor if needed
        pad_size = width * height - N
        if pad_size > 0:
            tensor = torch.cat([tensor, torch.zeros(pad_size)])
        
        # Reshape to 2D
        tensor_2d = tensor.reshape(height, width)
        
        # Save image
        plt.imshow(tensor_2d, cmap=cmap)
        plt.colorbar()
        plt.savefig(filename)
        plt.close()
        print(f"Image saved to {filename}")


    def interpret(
            self,
            inputs: Dict[str, torch.Tensor],
            generated_tokens: List[str],
            textual_prompt: str = "",
            attention_mask: Optional[torch.Tensor] = None, 
            pixel_values: Optional[torch.Tensor] = None,
            visual_tokens: int = 576,  # Common for many VLMs
            redistribute_flow: bool = False,
            target_token_idx: Optional[int] = None,
            start_layer=start_layer,
            start_layer_text=start_layer_text):
        

    # Dictionaries to store activations and gradients
        activations = {}
        gradients = {}

        # Forward hook to save activations
        def forward_hook(name):
            def hook(module, input, output):
                # Save activation and ensure it requires grad
                out = output.detach()
                out.requires_grad_(True)
                activations[name] = out
                return out
            return hook

        # Register hooks on all post_attention_layernorm layers
        for i, layer in enumerate(self.model.language_model.layers):
            layer_name = f"model.language_model.layers.{i}.post_attention_layernorm"
            layer.post_attention_layernorm.register_forward_hook(forward_hook(layer_name))

        # ===== Forward Pass =====
        outputs = self.model(**inputs)  # inputs = {'input_ids': ..., 'pixel_values': ...} depending on model
        logits_per_image = outputs.logits[:, -1, :]  # last token logits
        probs = logits_per_image.softmax(dim=-1)

        # ===== Define "pseudo-loss" as max probability for each sample =====
        # You can pick the max prob per sample (or any specific target index)
        max_probs = probs.max(dim=-1)[0]  # shape: (batch_size,)

        # For autograd, sum over batch to get a scalar
        pseudo_loss = max_probs.sum()

        # ===== Compute gradients using autograd =====
        for name, act in activations.items():
            grad = torch.autograd.grad(pseudo_loss, act, retain_graph=True)[0]
            gradients[name] = grad
        

        # =====Layer Relevance Extraction===

        local_relevance = self.compute_local_relevance(gradients, activations) # equation 8
        agrigate_gradient = self.aggregated_attention_gradient(gradients) # equaiton 9
        num_layers = len(local_relevance.keys())
        print(f"Number of layers with attention: {num_layers}")
        depth_weights = self.compute_depth_weights(num_layers, lambda_d=1.0) # equation 10
        adaptive_layer_weights = self.compute_adaptive_layer_weights(agrigate_gradient, depth_weights) # equation 11
        
        for layer_name, E_l in agrigate_gradient.items():
            print(f"Local relevance for layer {layer_name}: {E_l.shape}")  
            

            self.tensor_to_image( E_l.detach().cpu(), filename=f'local_relevance_{layer_name.replace(".", "_")}.png')

        


    
