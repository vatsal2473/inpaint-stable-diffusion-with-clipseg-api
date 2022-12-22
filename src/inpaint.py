import torch
import PIL

def stable_diffusion_inpaint(prompt, pipe, image_path, mask_path):
    # prompt = "gandhiji, black and white photo"
    image = PIL.Image.open(image_path).convert("RGB")
    mask_image = PIL.Image.open(mask_path).convert("RGB")

    image = image.resize((512, 512))
    mask_image = mask_image.resize((512, 512))

    guidance_scale=7.5
    num_samples = 3
    generator = torch.Generator(device="cuda").manual_seed(0) # change the seed to get different results

    images = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_samples,
    ).images

    return images