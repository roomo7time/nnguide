#!/bin/bash

# Loop through each cn value from config01 to config05
for cn in config01 config02 config03 config04 config05; do
  # Loop through each si value from 0 to 4
  for si in {0..4}; do
    # Run the Python script with the current cn and si values
    python main.py -cn "$cn" -si "$si"
  done
done