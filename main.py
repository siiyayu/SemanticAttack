from semantic_attack import SemanticAttack
from attn_processor import AttnProcessor
from utils import *

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
                           num_inference_steps_mask=10,
                           mask_threshold=0.5,
                           perturbation_budget=0.06,
                           attacking_step_size=0.07,
                           number_of_attacking_steps=1,
                           num_diffusion_steps=2
                           )

    model.generate_mask_using_forward_noise()
    immunized_latent = model.timestep_universal_gradient_updating()
