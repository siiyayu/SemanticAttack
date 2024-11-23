from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionImg2ImgPipeline, image_processor
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from attn_processor import AttnProcessor
from utils import *
from tqdm.auto import tqdm
from memory_profiler import profile


class SemanticAttack():
    def __init__(self,
                 image,
                 editing_prompt,
                 token, device,
                 attention_processor_class=AttnProcessor,
                 num_inference_steps_mask=50,
                 mask_threshold=0.5,
                 perturbation_budget=0.06,
                 attacking_step_size=0.01,
                 number_of_attacking_steps=100,
                 num_diffusion_steps=10) -> None:

        self.token = token
        self.editing_prompt = editing_prompt
        self.attention_processor_class = attention_processor_class
        # self.attention_maps = {}

        # constants (move it to the sem att function
        self.mask_threshold = mask_threshold
        self.perturbation_budget = perturbation_budget
        self.attacking_step_size = attacking_step_size
        self.number_of_attacking_steps = number_of_attacking_steps
        self.num_diffusion_steps = num_diffusion_steps

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
        self.init_latent = None

        self.loss = []

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

    def average_attention_maps(self):  # works
        '''
        Upsamples the attention maps and averages them across all layers.
        num_steps: int, the number of timesteps to average over.
        '''
        resolution_to_upsample_to = 64
        to_average = []
        for key, value in self.transformer_block_name_to_attention_processor_map.items():
            avg_attention_map = value.sum_attention_maps / value.counter
            res = avg_attention_map.shape[1]
            if res == resolution_to_upsample_to:
                to_average.append(avg_attention_map)
            else:
                to_average.append(self.upsample_attention(avg_attention_map, resolution_to_upsample_to))
        to_average = torch.cat(to_average, dim=0) # list of (16, 64, 64, 10) (num_heads * batch, res, res, num_tokens) -> (256, 64, 64, 10)

        averaged = to_average.sum(dim=0) / to_average.shape[0] # (256, 64, 64, 10) -> (64, 64, 10)
        # del to_average
        # del attention_map
        return averaged

    def denoise(self):
        '''
        Generates mask using sampling from the noisy state obtained with forward pass.
        '''
        # check_memory()

        # text embeddings
        text_embeddings = self.get_text_embeddings(prompt=self.token)
        # print(get_tensor_size(text_embeddings))

        with torch.no_grad():
            # timesteps
            self.scheduler.set_timesteps(self.num_inference_steps_mask, device=self.device)

            init_timestep = min(int(self.num_inference_steps_mask * self.strength), self.num_inference_steps_mask)
            t_start = max(self.num_inference_steps_mask - init_timestep, 0)

            timesteps = self.scheduler.timesteps[t_start:]
            num_inference_steps = self.num_inference_steps_mask - t_start

            # latents
            # init_latent = self.vae.encode(self.preprocessed_image).latent_dist.sample(self.generator)
            init_latent = self.get_init_latent()

            # print(get_tensor_size(init_latent))
            self.latent_shape = init_latent.shape
            init_latent = self.vae.config.scaling_factor * init_latent
            noise = torch.randn(init_latent.shape, generator=self.generator, device=self.device)
            latent_timestep = timesteps[:1]
            latent = self.scheduler.add_noise(init_latent, noise, latent_timestep)

            for t in tqdm(timesteps, total=len(timesteps)):
                latent_model_input = torch.cat([latent] * 2)  # cfg
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # print(get_tensor_size(latent_model_input))

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
        self.clean_attention_processors()
        self.denoise()
        attention_map = self.average_attention_maps()
        token_attention_map = attention_map[:, :, 1]
        masked_token_attention_map = token_attention_map > (token_attention_map.max() * self.mask_threshold)
        masked_token_attention_map = masked_token_attention_map.float()
        upsampled_masked_token_attention_map = F.interpolate(masked_token_attention_map.unsqueeze(0).unsqueeze(0),
                                                             size=(512, 512), mode='bicubic', align_corners=False)
        upsampled_masked_token_attention_map = upsampled_masked_token_attention_map.repeat(1, 3, 1, 1)

        # mask = masked_token_attention_map.unsqueeze(0).unsqueeze(0).repeat(1, self.latent_shape[1], 1, 1)
        self.mask = upsampled_masked_token_attention_map
        self.clean_attention_processors()

    def clean_attention_processors(self):
        for key, value in self.transformer_block_name_to_attention_processor_map.items():
            value.sum_attention_maps = None
            value.counter = 0
        # gc.collect()

    def get_text_embeddings(self, prompt):
        '''
        Gets text embedding with classifier-free guidance
        :return: text embeddings
        '''

        if self.text_embeddings is None:

            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                # max_length=self.tokenizer.model_max_length,
                max_length=10,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                # embeddings
                text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

                # classifier free guidance
                max_length = text_input.input_ids.shape[-1]
                uncond_input = self.tokenizer("", padding="max_length", max_length=max_length, return_tensors="pt")
                uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

                self.text_embeddings = text_embeddings

        else:
            text_embeddings = self.text_embeddings

        return text_embeddings

    def get_init_latent(self):
        if self.init_latent is None:
            init_latent = self.vae.encode(self.preprocessed_image).latent_dist.sample(self.generator)
            self.init_latent = init_latent
        else:
            init_latent = self.init_latent
        return init_latent

    def timestep_universal_gradient_updating(self):
        '''
        Performs universal gradient updating.
        '''

        text_embeddings = self.get_text_embeddings(prompt=self.token)
        # text_embeddings = self.text_embeddings

        self.scheduler.set_timesteps(self.num_diffusion_steps, device=self.device)

        diffusion_timesteps = self.scheduler.timesteps

        image_adv = self.preprocessed_image.clone()
        shape = image_adv.shape

        torch.autograd.set_detect_anomaly(True)

        delta = torch.zeros(shape).to(self.device)

        for attacking_step in tqdm(range(self.number_of_attacking_steps), total=self.number_of_attacking_steps,
                                   desc="Attacking steps"):

            accumulated_grad = torch.zeros(shape).to(self.device)  # Accumulate gradient across timesteps

            for t in tqdm(diffusion_timesteps, total=len(diffusion_timesteps),
                          desc=f"Attacking step {attacking_step + 1}"):
                image_adv_clone = image_adv.clone().detach().requires_grad_(True)
                masked_image_adv = self.mask * image_adv
                init_latent_adv = self.vae.encode(image_adv_clone).latent_dist.sample(self.generator)
                init_latent_adv = (self.vae.config.scaling_factor * init_latent_adv).to(self.device)


                noise = torch.randn(init_latent_adv.shape, generator=self.generator, device=self.device)
                noised_latent_adv = self.scheduler.add_noise(init_latent_adv, noise, t)

                latent_model_input = torch.cat([noised_latent_adv] * 2)  # cfg
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                masked_latent_model_input = latent_model_input
                noise_pred = self.unet(
                    masked_latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    # timestep_cond=None,
                    # cross_attention_kwargs=self.cross_attention_kwargs,
                    # added_cond_kwargs=added_cond_kwargs,
                    # return_dict=False,
                )[0]
                # del noise_pred

                # calculating loss
                attention_map = self.average_attention_maps()[:, :, 1]
                if image_adv_clone.grad is not None:
                    image_adv_clone.grad.zero_()
                loss = torch.norm(attention_map, p=1)
                self.loss.append(loss.item())

                loss.backward(retain_graph=True)

                accumulated_grad += image_adv_clone.grad
                self.clean_attention_processors()

            grad = accumulated_grad / len(diffusion_timesteps)
            delta = delta + self.attacking_step_size * torch.sign(grad)
            delta.clamp_(-self.perturbation_budget, self.perturbation_budget)
            image_adv = (image_adv - delta)  # Reset for next step
        return image_adv

    def register_custom_attention_processors(self):
        '''
        Registers a custom attention processor for each BasicTransformerBlock in the U-Net to extract attention maps after softmax.
        processor: class
        '''
        for name, module in self.unet.named_modules():
            if module.__class__.__name__ == 'BasicTransformerBlock':
                module.attn2.processor = self.attention_processor_class()
                self.transformer_block_name_to_attention_processor_map[name] = module.attn2.processor

    def show_attention_maps(self, directions, token, resolution):
        '''
        Shows averaged cross attention across timesteps, batch * n_heads, and layers
        with the same resolution attention maps for a given resolution and direction.
        direction: list, for example ["down", "up", "mid"]
        token: int, for example 1
        resolution: int for example 16
        '''
        filtered_attention_sums = []
        for key, value in self.transformer_block_name_to_attention_processor.items():
            dir = key.split("_")[0]
            res = value.sum_attention_maps.shape[1]
            if dir in directions and res == resolution:
                filtered_attention_sums.append(value.sum_attention_maps)
        filtered_attention_sums = torch.cat(filtered_attention_sums, dim=0)
        filtered_attention_sums = filtered_attention_sums.sum(dim=0) / filtered_attention_sums.shape[0]
        filtered_attention_sums = filtered_attention_sums[:, :, token]
        filtered_attention_sums = 255 * filtered_attention_sums / filtered_attention_sums.max()
        filtered_attention_sums = filtered_attention_sums.unsqueeze(-1).expand(*filtered_attention_sums.shape, 3)
        filtered_attention_sums = filtered_attention_sums.numpy().astype(np.uint8)
        return Image.fromarray(filtered_attention_sums).resize((256, 256))
