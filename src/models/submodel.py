import torch.nn as nn

def get_submodule(model, layer_name: str):
    """Traverse model by dotted name (e.g. 'language_model.layers.27.post_attention_layernorm')."""
    parts = layer_name.split(".")
    submod = model
    for p in parts:
        if p.isdigit():
            submod = submod[int(p)]
        else:
            submod = getattr(submod, p)
    return submod


class SubModelFromLayer(nn.Module):
    def __init__(self, full_model, layer_name: str):
        super().__init__()
        self.full_model = full_model
        self.layer_name = layer_name

        # find start index from layer_name if inside a ModuleList
        parts = layer_name.split(".")
        if "layers" in parts:
            layers_idx = parts.index("layers")
            layer_num = int(parts[layers_idx+1])
        else:
            raise ValueError("Layer name must be inside language_model.layers.N")

        # store everything after that layer
        self.remaining_layers = nn.ModuleList(full_model.language_model.layers[layer_num+1:])
        self.final_norm = full_model.language_model.norm
        self.lm_head = full_model.lm_head

    def forward(self, hidden_states):
        # We assume `hidden_states` already comes from the specified submodule
        for layer in self.remaining_layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits
