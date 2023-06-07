import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config

import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))
batch_size = 4
guidance_scale = 15.0
prompt = "a shark"

latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[prompt] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
)

# Example of saving the latents as meshes.
from shap_e.util.notebooks import decode_latent_mesh

os.makedirs('./output', exist_ok = False)

for i, latent in enumerate(latents):
    t = decode_latent_mesh(xm, latent).tri_mesh()
    with open(f'./output/mesh_{i}.ply', 'wb') as f:
        t.write_ply(f)
    with open(f'./output/mesh_{i}.obj', 'w') as f:
        t.write_obj(f)


'''
// .obj file and .ply file successfully generated, but maybe you get an error below
// this error is also found on issue section of shap-e GitHub page (https://github.com/openai/shap-e/issues/56)
Bin size was too small in the coarse rasterization phase. This caused an overflow, 
meaning output may be incomplete. To solve, try increasing max_faces_per_bin / max_points_per_bin, decreasing bin_size, 
or setting bin_size to 0 to use the naive rasterization

'''