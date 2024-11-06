import torch
from diffusers import StableDiffusionPipeline

attention_maps = {}
inputs = []

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def capture_attention(name):
    def hook(module, input, output):
      if len(input) < 2:
        print("self attention")
      else:
        print("cross attention")
      # Use the provided name as the key in the attention_maps dictionary
      # if name not in attention_maps:
      #   attention_maps[name] = []
      inputs.append(input)
      #   # `sim` is the unnormalized attention score, `attn` is the softmaxed version
      # with torch.no_grad():
      #   h = module.heads
      #   q = module.to_q(input[0])  # Calculate query
      #   context = default(input[1], input[0])  # Default to `x` if `context` is None
      #   k = to_k(context)  # Calculate key

      #   # Reshape q and k for attention calculation
      #   q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)
      #   k = rearrange(k, 'b n (h d) -> (b h) n d', h=h)

      #   # Compute raw attention scores and apply scaling
      #   sim = torch.einsum('b i d, b j d -> b i j', q, k) * module.scale
      #   attn = sim.softmax(dim=-1)  # Softmax to get attention map

      #   # Store the attention map in the dictionary
      #   if name not in attention_maps:
      #     attention_maps[name] = []
      #   attention_maps[name].append(attn.cpu())
      #   print(f"Hook registered for layer: {name}")

    return hook


        # h = self.heads

        # q = self.to_q(x)
        # context = default(context, x)
        # k = self.to_k(context)
        # v = self.to_v(context)

        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # if exists(mask):
        #     mask = rearrange(mask, 'b ... -> b (...)')
        #     max_neg_value = -torch.finfo(sim.dtype).max
        #     mask = repeat(mask, 'b j -> (b h) () j', h=h)
        #     sim.masked_fill_(~mask, max_neg_value)

        # # attention, what we cannot get enough of
        # attn = sim.softmax(dim=-1)

        # out = einsum('b i j, b j d -> b i d', attn, v)
        # out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # return self.to_out(out)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load the Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cpu")

    modules = []
    # Register the hook for each cross-attention layer in the U-Net
    for name, module in pipe.unet.named_modules():
        # print(name)
        if name.split(".")[-1] == "attn2":
            print(name)
            modules.append(module)
            # if isinstance(module, CrossAttention):
            module.register_forward_hook(capture_attention(name))
    prompt = "A beautiful painting of a sunset over a mountain landscape"
    with torch.no_grad():
      images = pipe(prompt, num_inference_steps=50)