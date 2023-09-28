#!/bin/bash

# model_name choices
model_names=("resnet50-supcon" "resnet50-react" "regnet-y-16gf-swag-e2e-v1" "vit-b16-swag-e2e-v1" "mobilenet-v2")

# id_data_name choices
id_data_names=("imagenet1k" "imagenet1k-v2-a" "imagenet1k-v2-b" "imagenet1k-v2-c")

# ood_data_name choices
ood_data_names=("inaturalist" "sun" "places" "textures" "openimage-o")

# Iterate through all combinations
for model in "${model_names[@]}"; do
    for id_data in "${id_data_names[@]}"; do
        for ood_data in "${ood_data_names[@]}"; do
            # Running the main.py with the current combination
            python main.py --model_name "$model" --id_data_name "$id_data" --ood_data_name "$ood_data"
        done
    done
done