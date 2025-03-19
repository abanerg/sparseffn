import torch
class Inference():
    """
    Artifact of data from an inference ran through on a prompt.

    """
    # In general, the inferences stored in new_mmlu_tests have all fields (input, post silu, post up, post down)
    # May add leaner inferences that only have for example input and post up * post silu (two vectors per step instead of 4)
    set_types = ["MMLU",
                 ]
    def __init__(self, set, index, prompt, real_answer, output, layers, logits):
        self.prompt = prompt
        self.real_answer = real_answer
        self.set = set
        self.index = index
        self.output = output
        self.n_layers = len(layers.keys())
        self.layers = layers # layer -> step -> inp, silu, up, down
        self.logits = logits
    def available_fields(self):
        return list(self.layers[0][0].keys()) # guaranteed to have at least 1 layer, 1 step
    def steps_at_layer(self, layer):
        if layer < 0 or layer >= self.n_layers:
            raise ValueError(f"Layer index out {layer} of range for inference on {self.n_layers} layers")
        else:
            return self.layers[layer]
    def get_inputs_at_layer(self, layer):
        return self.get_vectors_at_layer(layer, "input")
    def get_input_at_layer_step(self, layer, step):
        """
        Try to avoid step 0
        """
        return self.get_vector_at_layer_step(layer, step, "input")
    def get_post_silus_at_layer(self, layer):
        return self.get_vectors_at_layer(layer, "post silu")
    def get_post_silu_at_layer_step(self, layer, step):
        """
        Try to avoid step 0
        """
        return self.get_vector_at_layer_step(layer, step, "post silu")
    def get_post_up_projs_at_layer(self, layer):
        return self.get_vectors_at_layer(layer, "post up proj")
    def get_post_up_proj_at_layer_step(self, layer, step):
        """
        Try to avoid step 0
        """
        return self.get_vector_at_layer_step(layer, step, "post up proj")
    def get_post_down_projs_at_layer(self, layer): #TODO: if we add leaner inferences, update to check if even exists
        return self.get_vectors_at_layer(layer, "post down proj")
    def get_post_down_proj_at_layer_step(self, layer, step):
        """
        Try to avoid step 0
        """
        return self.get_vector_at_layer_step(layer, step, "post down proj")
    def get_hs_at_layer(self, layer):
        silu = self.get_post_silus_at_layer(layer)
        up = self.get_post_up_projs_at_layer(layer)
        return [torch.mul(silu[i], up[i]) for i in range(len(silu))]
    def get_h_at_layer_step(self, layer, step):
        """
        Try to avoid step 0
        """
        return torch.mul(self.get_vector_at_layer_step(layer, step, "post silu"),
                         self.get_vector_at_layer_step(layer, step, "post up proj"))
    def get_vectors_at_layer(self, layer, field):
        if layer < 0 or layer >= self.n_layers:
            raise ValueError(f"Layer index {layer} out of range for inference on {self.n_layers} layers")
        return [self.layers[layer][j][field] for j in range(1, len(self.layers[0]))]
    def get_vector_at_layer_step(self, layer, step, field):
        if layer < 0 or layer >= self.n_layers:
            raise ValueError(f"Layer index {layer} out of range for inference on {self.n_layers} layers")
        if step < 0 or step >= len(self.layers[layer]):
            raise ValueError(f"Step index {step} out of range for inference on {self.n_layers} steps")
        else:
            return self.layers[layer][j][field]
    def get_product_training_data(self):
        """
        Of the form of a list of dicts (indices correspond to layers), keys are "inputs", "products" pointing
        to list going over all (minus step 0)
        """
        return [self.get_product_training_data_at_layer(i) for i in range(self.n_layers)]
    def get_product_training_data_at_layer(self, layer):
        if layer < 0 or layer >= self.n_layers:
            raise ValueError(f"Layer index out {layer} of range for inference on {self.n_layers} layers")
        else:
            inputs = self.get_inputs_at_layer(layer)
            return {"inputs": inputs, "products": self.get_hs_at_layer(layer), "num": len(inputs)}
    def get_response(self):
        """
        Only use this to get response from training data (don't access output field manually)
        """
        try:
            return self.output
        except:
            return self.response