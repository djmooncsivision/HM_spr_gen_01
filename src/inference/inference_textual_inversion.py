from diffusers import StableDiffusionPipeline
import torch
import datetime

def main():
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    repo_id = "DJMOON/spr_ti_01"
    prompt = "<SPR3_BG0G46E(DEHG13598)_SGACUD(0.7t)_SABC1470(1.1t)_A365.0(2.5t)>"

    pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
    pipeline.load_textual_inversion(repo_id)
    image = pipeline(prompt, num_inference_steps=50).images[0]
    image.save(f"inference_result_{timestamp}.png")
    

if __name__ == "__main__":
    main()