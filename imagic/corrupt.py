from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
from diffusers import DDIMScheduler
from PIL import Image

from imagic_stable_diffusion_v2 import CorruptImagicStableDiffusionPipeline


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--init_image", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--corrupt_image", type=Path, required=True)
    parser.add_argument("--corrupt_prompt", type=str, required=True)

    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.7, nargs="+")

    parser.add_argument("--output_dir", type=Path, default="./imagic_corrupt/")
    parser.add_argument("--device", type=torch.device, default="cuda")
    parser.add_argument("--num_samples", type=int, default=1)

    args = parser.parse_args()
    return args


def get_image(image_path, height=512, width=512):
    image = Image.open(image_path)
    image = image.resize((height, width))
    return image


def main(args):
    pipe = CorruptImagicStableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        use_auth_token=True,
        scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            clip_sample=False, set_alpha_to_one=False)
    ).to(args.device)
    generator = torch.Generator(args.device).manual_seed(args.seed)

    init_image = get_image(args.init_image, height=args.height, width=args.width)
    corrupt_image = get_image(args.corrupt_image, height=args.height, width=args.width)

    res = pipe.train(
        args.prompt,
        image=init_image,
        generator=generator,
        height=args.height, width=args.width,
        embedding_learning_rate=1e-3, text_embedding_optimization_steps=500,
        diffusion_model_learning_rate=2e-6, model_fine_tuning_optimization_steps=1000,
        corrupt_prompt=args.corrupt_prompt,
        corrupt_image=corrupt_image,
        corrupt_optimization_steps=500,
        corrupt_learning_rate=2e-6)
        # corrupt_learning_rate=4e-6,
        # corrupt_optimization_steps=700,)

    for alpha in args.alpha:
        for i in range(args.num_samples):
            res = pipe(alpha=alpha, guidance_scale=7.5, num_inference_steps=50,
                       height=args.height, width=args.width)
            image = res.images[0]
            image.save(Path(args.output_dir, f'result_alpha_{alpha}_{i}.png'))


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args) 
