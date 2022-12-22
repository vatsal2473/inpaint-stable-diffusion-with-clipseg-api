from diffusers import StableDiffusionInpaintPipeline
import torch
from models.clipseg.models.clipseg import CLIPDensePredT


def load_stable_diffusion_model():
    pipe = None
    device = "cuda"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16", 
        torch_dtype=torch.float16,
        use_auth_token=True
    ).to(device)

    return pipe

def load_clipseg():
    global clipseg_model
    clipseg_model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    # clipseg_model.eval();
    clipseg_model.load_state_dict(torch.load('models/clipseg/weights/rd64-uni.pth', map_location=torch.device('cuda')), strict=False)
    return clipseg_model

