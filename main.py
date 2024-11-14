from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionImg2ImgPipeline, image_processor
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from diffusers.models.attention import Attention
from typing import Optional
import requests
from io import BytesIO
from tqdm.auto import tqdm


class AttnProcessor:
    """
    Default processor for performing attention-related computations.
    """
    def __init__(self):
        # Initialize an empty list to store attention maps for each call
        self.attention_maps = []
        # self.hidden_states = []
        # self.encoder_hidden_states = []
        # self.attention_mask = []

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        res = int(np.sqrt(attention_probs.shape[1]))
        num_tokens = attention_probs.shape[-1]
        self.attention_maps.append(attention_probs.view(-1, res, res, num_tokens))
        # self.hidden_states.append(hidden_states)
        # self.encoder_hidden_states.append(encoder_hidden_states)
        # self.attention_mask.append(attention_mask)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def load_image(url, size=None):
    response = requests.get(url,timeout=1)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    if size is not None:
        img = img.resize(size)
    return img


class SemanticAttack():
    def __init__(self,
                 image,
                 editing_prompt,
                 token, device,
                 attention_processor_class=AttnProcessor,
                 num_inference_steps_mask=50) -> None:

        self.token = token
        self.editing_prompt = editing_prompt
        self.attention_processor_class = attention_processor_class
        # self.attention_maps = {}

        # constants
        self.mask_threshold = 0.5
        self.perturbation_budget = 0.06
        self.attacking_step_size = 0.07
        self.number_of_attacking_steps = 10
        self.num_diffusion_steps = 10

        self.strength = 0.5
        self.guidance_scale = 6
        self.num_inference_steps_mask = num_inference_steps_mask

        # Initializing block for the Stable Diffusion Model
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)
        self.tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder",
                                                          use_safetensors=True)
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet",
                                                         use_safetensors=True)
        self.scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

        # Let's also save pipeline to use it's functions
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                                                       safety_checker=None,
                                                                       requires_safety_checker=False)

        # choosing device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print("Using:", self.device)

        self.generator = torch.Generator(device=self.device)
        torch.manual_seed(0)

        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        self.unet.to(self.device)

        self.image = image
        self.preprocessed_image = self.pipeline.image_processor.preprocess(image).to(
            self.device)  # maybe VAE image processor?

        self.transformer_block_name_to_attention_processor_map = {}
        self.register_custom_attention_processors()

        self.mask = None
        self.latent_shape = None
        self.text_embeddings = None

    def upsample_attention(self, attention_map, target_res):  # works
        '''
        Upsamples attention map to a specific shape
        attention_map: torch.Tensor of size: [batch_size * n_heads, res, res, n_tokens]
        shape: tuple,
        return: torch.Tensor of size: [batch_size * n_heads, target_res, target_res, n_tokens]
        '''
        batch_size_n_heads, _, res, n_tokens = attention_map.shape
        # [batch_size * n_heads, target_res, target_res, n_tokens] -> [batch_size * n_heads * n_tokens, 1, target_res, target_res]
        attention_map = attention_map.permute(0, 3, 1, 2).reshape(-1, 1, res, res)
        attention_map = F.interpolate(attention_map, size=(target_res, target_res), mode='bicubic', align_corners=False)
        # [batch_size * n_heads * n_tokens, 1, target_res, target_res] -> [batch_size * n_heads, target_res, target_res, n_tokens]
        attention_map = attention_map.reshape(batch_size_n_heads, n_tokens, target_res, target_res).permute(0, 2, 3, 1)
        return attention_map

    def average_attention_maps(self, num_steps=1):  # works
        '''
        Upsamples the attention maps and averages them across all layers.
        num_steps: int, the number of timesteps to average over.
        '''
        resolution_to_upsample_to = 64
        to_average = []
        for key, value in self.transformer_block_name_to_attention_processor_map.items():
            attention_map = value.attention_maps[-num_steps:].copy()
            attention_map = sum(attention_map) / len(attention_map)
            res = attention_map.shape[1]
            if res == resolution_to_upsample_to:
                to_average.append(attention_map)
            else:
                to_average.append(self.upsample_attention(attention_map, resolution_to_upsample_to))
        to_average = torch.cat(to_average, dim=0)
        averaged = to_average.sum(dim=0) / to_average.shape[0]
        del to_average
        del attention_map
        return averaged

    def denoise(self):
        '''
        Generates mask using sampling from the noisy state obtained with forward pass.
        '''

        # text embeddings
        text_embeddings = self.get_text_embeddings(prompt=self.token)

        with torch.no_grad():
            # timesteps
            self.scheduler.set_timesteps(self.num_inference_steps_mask, device=self.device)

            init_timestep = min(int(self.num_inference_steps_mask * self.strength), self.num_inference_steps_mask)
            t_start = max(self.num_inference_steps_mask - init_timestep, 0)

            timesteps = self.scheduler.timesteps[t_start:]
            num_inference_steps = self.num_inference_steps_mask - t_start

            # latents
            init_latent = self.vae.encode(self.preprocessed_image).latent_dist.sample(self.generator)
            self.latent_shape = init_latent.shape
            init_latent = self.vae.config.scaling_factor * init_latent
            noise = torch.randn(init_latent.shape, generator=self.generator, device=self.device)
            latent_timestep = timesteps[:1]
            latent = self.scheduler.add_noise(init_latent, noise, latent_timestep)

            for t in tqdm(timesteps, total=len(timesteps)):
                latent_model_input = torch.cat([latent] * 2)  # cfg
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    # timestep_cond=None,
                    # cross_attention_kwargs=self.cross_attention_kwargs,
                    # added_cond_kwargs=added_cond_kwargs,
                    # return_dict=False,
                )[0]

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latent = self.scheduler.step(noise_pred, t, latent, return_dict=False)[0]
        return

    def generate_mask_using_forward_noise(self):
        self.denoise()
        attention_map = model.average_attention_maps(num_steps=50)
        token_attention_map = attention_map[:, :, 1]
        masked_token_attention_map = token_attention_map * (token_attention_map > token_attention_map.max() * 0.5)
        mask = masked_token_attention_map.unsqueeze(0).unsqueeze(0).repeat(1, self.latent_shape[1], 1, 1)
        self.mask = mask
        self.clean_attention_processors()

    def clean_attention_processors(self):
        for key, value in self.transformer_block_name_to_attention_processor_map.items():
            value.attention_maps = []


    def get_text_embeddings(self, prompt):
        '''
        Gets text embedding with classifier-free guidance
        :return: text embeddings
        '''
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            # max_length=self.tokenizer.model_max_length,
            max_length=10,
            truncation=True,
            return_tensors="pt",
        )

        if self.text_embeddings is None:

            # embeddings
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

            # classifier free guidance
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer("", padding="max_length", max_length=max_length, return_tensors="pt")
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            self.text_embeddings = text_embeddings

            return text_embeddings

        else:
            return self.text_embeddings

    def print_graph(self, fn, indent=0):
        if fn is None:
            return
        print("  " * indent + str(fn))
        for next_fn, _ in fn.next_functions:
            self.print_graph(next_fn, indent + 1)

    def timestep_universal_gradient_updating(self):
        '''
        Performs universal gradient updating.
        '''

        text_embeddings = self.get_text_embeddings(prompt=self.token)
        # text_embeddings = self.text_embeddings

        self.scheduler.set_timesteps(self.num_diffusion_steps, device=self.device)

        diffusion_timesteps = self.scheduler.timesteps

        with torch.no_grad():
            init_latent = self.vae.encode(self.preprocessed_image).latent_dist.sample(self.generator)
            init_latent = self.vae.config.scaling_factor * init_latent

        delta = torch.zeros(init_latent.shape)

        init_latent_adv = init_latent.clone().detach().requires_grad_(True)

        for attacking_step in tqdm(range(self.number_of_attacking_steps), total=self.number_of_attacking_steps,
                                   desc="Attacking steps"):
            accumulated_grad = torch.zeros_like(init_latent_adv)  # Accumulate gradient across timesteps
            for t in tqdm(diffusion_timesteps, total=len(diffusion_timesteps),
                          desc=f"Attacking step {attacking_step + 1}"):
                with torch.autograd.profiler.profile() as prof:
                    noise = torch.randn(init_latent_adv.shape, generator=self.generator, device=self.device)
                    noised_adv_latent = self.scheduler.add_noise(init_latent_adv, noise, t)

                    latent_model_input = torch.cat([noised_adv_latent] * 2)  # cfg
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    masked_latent_model_input = self.mask * latent_model_input
                    noise_pred = self.unet(
                        masked_latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        # timestep_cond=None,
                        # cross_attention_kwargs=self.cross_attention_kwargs,
                        # added_cond_kwargs=added_cond_kwargs,
                        # return_dict=False,
                    )[0]
                    del noise_pred

                    # calculating loss
                    attention_map = self.average_attention_maps(num_steps=1)[:, :, 1]
                    if init_latent_adv.grad is not None:
                        init_latent_adv.grad.zero_()
                    loss = torch.norm(attention_map, p=1)
                    loss.backward(retain_graph=True)

                    ##
                    # self.print_graph(attention_map)
                    ##
                    with torch.no_grad():
                        accumulated_grad += init_latent_adv.grad
                # print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))
            self.clean_attention_processors()

            grad = accumulated_grad / len(diffusion_timesteps)
            delta = delta + self.attacking_step_size * torch.sign(grad)
            delta.clamp_(-self.perturbation_budget, self.perturbation_budget)
            with torch.no_grad():
                init_latent_adv = (init_latent - delta).detach().requires_grad_(True)  # Reset for next step

        return init_latent_adv

    def register_custom_attention_processors(self):
        '''
        Registers a custom attention processor for each BasicTransformerBlock in the U-Net to extract attention maps after softmax.
        processor: class
        '''
        for name, module in self.unet.named_modules():
            if module.__class__.__name__ == 'BasicTransformerBlock':
                module.attn2.processor = self.attention_processor_class()
                self.transformer_block_name_to_attention_processor_map[name] = module.attn2.processor

    def show_cross_attention_maps(self, direction=["down", "up"], token=8, resolution=16):
        '''
        Shows averaged cross attention across timesteps, batch * n_heads, and layers
        with the same resolution attention maps for a given resolution and direction.
        direction: ["down", "up"]
        token: int
        resolution: int
        '''
        attention_maps = []
        for key, value in self.transformer_block_name_to_attention_processor.items():
            dir = key.split("_")[0]
            res = value.attention_maps[0].shape[1]
            if dir in direction and res == resolution:
                for item in value.attention_maps:
                    attention_maps.append(item)
        attention_maps = torch.cat(attention_maps, dim=0)
        attention_maps = attention_maps.sum(dim=0) / attention_maps.shape[0]
        attention_maps = attention_maps[:, :, token]
        attention_maps = 255 * attention_maps / attention_maps.max()
        attention_maps = attention_maps.unsqueeze(-1).expand(*attention_maps.shape, 3)
        attention_maps = attention_maps.numpy().astype(np.uint8)
        return Image.fromarray(attention_maps).resize((256, 256))
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_image = load_image('https://images.pexels.com/photos/8306128/pexels-photo-8306128.jpeg', size=(512, 512))
    device = "cpu"
    model = SemanticAttack(image=input_image,
                           editing_prompt="cat on the grass",
                           token='dog',
                           device="cpu",
                           attention_processor_class=AttnProcessor,
                           #  pipeline_class=StableDiffusionImg2ImgPipeline,
                           #  scheduler_class=DDIMScheduler,
                           num_inference_steps_mask=50)
    model.generate_mask_using_forward_noise()
    immunized_latent = model.timestep_universal_gradient_updating()
