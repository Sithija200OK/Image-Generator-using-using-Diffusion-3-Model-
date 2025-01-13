import torch
from diffusers import StableDiffusionPipeline
import gradio as gr
def image_generation(prompt):
    device = "cpu"  # Specify CPU as the device

    # Load the pipeline for CPU
    pipeline = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2",  # Adjusted to a valid pipeline name
        torch_dtype=torch.float32  # Use float32 for CPU
    )

    # Explicitly set the pipeline to CPU
    pipeline.to(device)

    # Generate the image
    image = pipeline(
        prompt=prompt,
        negative_prompt="blurred, ugly, watermark",
        num_inference_steps=40,  # Corrected argument
        height=1024,  # Fixed typo
        width=1024,  # Fixed typo
        guidance_scale=9.0
    ).images[0]

    # Display the image
    return image

# Call the function
#image_generation("magician cat doing spell")

interface= gr.Interface(
    fn=image_generation,
    inputs=gr.Textbox(lines=2,placeholder="Enter Your Prompt..."),
    outputs= gr.Image(type="pil"),
    title="@Sithija's Image Generator(using Diffusion 3 Model)"
)

interface.launch()
