#!/bin/bash

GPU=0

noise_settings=("none" "additive" "multiplicative")
for noise in "${noise_settings[@]}"; do
    for i in {1..5}; do
        echo "Running with --noise_setting: $noise (Run $i)"
        python /NSDE/experiment.py --gpu $GPU --noise_setting $noise
    done
done

noise_settings=("dropout_TTN" "dropout")
p_values=(1e-5 1e-4 1e-3 1e-2 0.1 0.2 0.3 0.4 0.5)

for noise in "${noise_settings[@]}"; do
    for p in "${p_values[@]}"; do
        for i in {1..5}; do
            echo "Running with --noise_setting: $noise, --p: $p (Run $i)"
            python /NSDE/experiment.py --gpu $GPU --noise_setting $noise --p $p
        done
    done
done