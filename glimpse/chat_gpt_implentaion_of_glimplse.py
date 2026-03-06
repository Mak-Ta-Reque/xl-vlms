import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForVision2Seq, AutoProcessor
import matplotlib.pyplot as plt

class GLIMPSE:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-2B-Instruct", device=None):
        self.device = device or os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, torch_dtype=torch.float16,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.model.eval()

    def forward(self, image, prompt, max_new_tokens=30):
        # Process inputs
        inputs = self.processor(prompt, images=image, return_tensors="pt").to(self.device)

        # Forward with attentions
        outputs = self.model(**inputs, output_attentions=True, return_dict=True)
        return inputs, outputs

    def compute_gradients(self, outputs, attns, gen_ids, gen_pos):
        """
        Compute ∂logit/∂A for each attention head.
        attns: list of [B,H,N,N] per layer
        gen_ids: [T] tensor of generated token ids
        gen_pos: [T] tensor of their positions in sequence
        """
        logits = outputs.logits  # [B,N,V]
        grads, attn_copy = [], []

        for l, A_l in enumerate(attns):  # A_l: [B,H,N,N]
            A_l.requires_grad_(True)
            grad_accum = torch.zeros_like(A_l)

            for pos, tok_id in zip(gen_pos, gen_ids):
                logit = logits[0, pos, tok_id]
                grad = torch.autograd.grad(
                    outputs=logit, inputs=A_l,
                    retain_graph=True, allow_unused=True
                )[0]
                if grad is not None:
                    grad_accum += grad

            grads.append(grad_accum)
            attn_copy.append(A_l)

        return grads, attn_copy

    def fuse_heads(self, g_lh, a_lh, lambda_head=1.0):
        """
        Eq. 5–8: Gradient × Attention fusion across heads
        g_lh, a_lh: [B,H,N,N]
        """
        G = F.relu(g_lh) * a_lh  # Eq. 5

        # Head importance scores
        num = G.sum(dim=(-2, -1))                # [B,H]
        den = F.relu(g_lh).sum(dim=(-2, -1)) + 1e-8
        head_scores = num / den

        w = F.softmax(head_scores / lambda_head, dim=1)  # [B,H]
        E = (G * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # [B,N,N]

        # Row normalize
        E = E / (E.sum(dim=-1, keepdim=True) + 1e-8)
        return E

    def adaptive_layer_weights(self, grads, lambda_depth=0.1):
        """
        Eq. 9–11: α_l = g_l * s_l / Σ g_k * s_k
        grads: list of [B,H,N,N]
        """
        L = len(grads)
        g = torch.tensor([gl.abs().sum().item() for gl in grads], device=self.device)

        depth = torch.arange(1, L+1, device=self.device, dtype=torch.float32)
        s = torch.exp(lambda_depth * depth)
        s = s / (s.sum() + 1e-8)

        alpha = (g * s)
        alpha = alpha / (alpha.sum() + 1e-8)
        return alpha

    def propagate(self, E_list, alpha):
        """
        Eq. 12–14: Sequential layer propagation
        """
        N = E_list[0].shape[-1]
        R = torch.eye(N, device=self.device)

        for E, a in zip(E_list, alpha):
            L_l = torch.eye(N, device=self.device) + a * E
            R = R + L_l @ R
        return R

    def token_weights(self, R, logits, gen_ids, Y_idx, V_idx, P_idx):
        """
        Eq. 15–20: Compute a_t, v_t, p_t, β, γ
        """
        # [T,N]
        R_Y = R[Y_idx, :]

        a_t = R_Y[:, P_idx].mean(dim=1)  # prompt alignment
        v_t = R_Y[:, V_idx].mean(dim=1)  # visual alignment

        probs = F.softmax(logits[0, Y_idx, :], dim=-1)  # [T,V]
        p_t = probs[torch.arange(len(Y_idx), device=self.device), gen_ids]

        beta_V = (p_t * a_t); beta_V /= (beta_V.sum() + 1e-8)
        beta_P = (p_t * v_t); beta_P /= (beta_P.sum() + 1e-8)
        gamma = beta_V * beta_P

        return beta_V, beta_P, gamma

    def holistic_maps(self, R, Y_idx, V_idx, P_idx, beta_V, beta_P):
        """
        Eq. 23: Aggregate over tokens
        """
        R_Y = R[Y_idx, :]
        R_V = (beta_V.unsqueeze(1) * R_Y[:, V_idx]).sum(dim=0)  # visual saliency
        R_P = (beta_P.unsqueeze(1) * R_Y[:, P_idx]).sum(dim=0)  # prompt saliency
        return R_V, R_P

    def explain(self, image, prompt, gen_ids, gen_pos,
                V_idx, P_idx, Y_idx):
        """
        Run full GLIMPSE pipeline.
        gen_ids: list/tensor of generated token IDs
        gen_pos: their positions in sequence
        V_idx, P_idx, Y_idx: indices for visual/prompt/generated tokens
        """
        inputs, outputs = self.forward(image, prompt)
        attns = outputs.attentions

        grads, attn_copy = self.compute_gradients(outputs, attns, gen_ids, gen_pos)
        E_list = [self.fuse_heads(g, a) for g,a in zip(grads, attn_copy)]
        alpha = self.adaptive_layer_weights(grads)
        R = self.propagate(E_list, alpha)

        beta_V, beta_P, gamma = self.token_weights(
            R, outputs.logits, gen_ids, Y_idx, V_idx, P_idx
        )
        R_V, R_P = self.holistic_maps(R, Y_idx, V_idx, P_idx, beta_V, beta_P)

        return {
            "R": R,
            "beta_V": beta_V,
            "beta_P": beta_P,
            "gamma": gamma,
            "R_V": R_V,
            "R_P": R_P
        }
from PIL import Image

glimpse = GLIMPSE()

# Example inputs
image = Image.open("dog.jpg")
prompt = "What is in the picture?"

# Suppose the model generated "a dog"
gen_tokens = glimpse.processor.tokenizer(" a dog").input_ids[1:]  # skip BOS
gen_ids = torch.tensor(gen_tokens, device=glimpse.device)
gen_pos = torch.arange(start=10, end=10+len(gen_tokens), device=glimpse.device)  # toy positions

# Fake modality splits (adjust based on your tokenizer/model internals)
V_idx = torch.arange(0, 576, device=glimpse.device)          # 24x24 visual tokens
P_idx = torch.arange(576, 600, device=glimpse.device)        # prompt tokens
Y_idx = torch.arange(600, 600+len(gen_ids), device=glimpse.device)

results = glimpse.explain(image, prompt, gen_ids, gen_pos, V_idx, P_idx, Y_idx)

print("Cross-modal token relevance γ:", results["gamma"])
print("Visual saliency map size:", results["R_V"].shape)
