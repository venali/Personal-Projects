import gradio as gr
import numpy as np
import random
from diffusers import DiffusionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    torch.cuda.max_memory_allocated(device=device)
    pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)
else: 
    pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True)
    pipe = pipe.to(device)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

def infer():

    seed = random.randint(0, MAX_SEED)
        
    generator = torch.Generator().manual_seed(seed)
    
    image = pipe(
        prompt = "CUTE WHITE SHIBA INU", 
        negative_prompt = "",
        generator = generator,
        guidance_scale=0.0,
        num_inference_steps=1
    ).images[0] 
    
    return image

css="""
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

if torch.cuda.is_available():
    power_device = "GPU"
else:
    power_device = "CPU"

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""
        # Generate White Shibu ^^
        Currently running on {power_device}.
        """)
        
        with gr.Row():
                        
            run_button = gr.Button("Run", scale=0)
        
        result = gr.Image(label="Result", show_label=False)

    run_button.click(
        fn = infer,
        outputs = [result]
    )

demo.queue().launch()