from semantic_attack import SemanticAttack
from attn_processor import AttnProcessor
from utils import *

if __name__ == '__main__':
    input_image = load_image('https://images.pexels.com/photos/8306128/pexels-photo-8306128.jpeg', size=(512, 512))
    device = "cpu"
    model = SemanticAttack(image=input_image,
                           editing_prompt="cat on the grass",
                           token='dog',
                           device="cuda",
                           attention_processor_class=AttnProcessor,
                           #  pipeline_class=StableDiffusionImg2ImgPipeline,
                           #  scheduler_class=DDIMScheduler,
                           num_inference_steps_mask=50,
                           mask_threshold=0.5,
                           perturbation_budget=0.06,
                           attacking_step_size=0.0001,
                           number_of_attacking_steps=100,
                           num_diffusion_steps=10
                           )

    model.generate_mask_using_forward_noise()
    immunized_image_tensor = model.timestep_universal_gradient_updating()
    immunized_image = \
    model.pipeline.image_processor.postprocess(immunized_image_tensor, output_type='pil', do_denormalize=[True])[0]