#!/bin/bash

python3.9 imagic.py \
  --init_image="./test_case/1/test.png" \
  --prompt="A photo of a dog sitting in the library, camera shooting with near focus" \
  --output_dir="./imagic/1/" \
  --height=256 \
  --width=256 \
  --num_samples=5 \
  --alpha 0.6 \

python3.9 imagic.py \
  --init_image="./test_case/2/test.png" \
  --prompt="A photo of a dog sitting in the library, camera shooting with far focus" \
  --output_dir="./imagic/2/" \
  --height=256 \
  --width=256 \
  --num_samples=5 \
  --alpha 0.6 \

python3.9 imagic.py \
  --init_image="./test_case/3/test.png" \
  --prompt="A photo of a teddy bear sitting on the bed, camera shooting with near focus" \
  --output_dir="./imagic/3/" \
  --height=256 \
  --width=256 \
  --num_samples=5 \
  --alpha 0.5 \

python3.9 imagic.py \
  --init_image="./test_case/4/slight_test.png" \
  --prompt="A photo of a panda standing in a forest covered in slight fog" \
  --output_dir="./imagic/4_slight/" \
  --height=256 \
  --width=256 \
  --num_samples=5 \
  --alpha 0.7 \

python3.9 imagic.py \
  --init_image="./test_case/4/heavy_test.png" \
  --prompt="A photo of a panda standing in a forest covered in heavy fog" \
  --output_dir="./imagic/4_heavy/" \
  --height=256 \
  --width=256 \
  --num_samples=5 \
  --alpha 0.7 \
