#!/bin/sh
echo ""
echo "                  ------------------------------                   "
echo "                 |                              |                  "
echo "-----------------  Model Compression Experiments ------------------"
echo "                 |                              |                  "
echo "                  ------------------------------                   " 
echo ""
echo "-------------------------------------------------------------------"
# Info:
# ----
# - run.py appends new entries into experiment.log
# ------------------------------------------------
# Random initialization experiments
for i in 1; do
	for depth in 32 56 110; do
		for lr in 1e-3; do #1e-3 for 32 56 110 - 1e-4 for 50
			echo "Exp #"$i" : Training variant = {depth: "$depth", lr: "$lr", random_init}"
			python run.py --depth $depth --lr $lr --optimizer adam --batch_size 128 --epochs 400 --history_interval 10 --datasets CIFAR-10 --significance_threshold 1e-4 --compression_modes spectral_norm --l1_penalties 1e-4 --modes_to_compare det det_corr min_eig min_eig_real weight spectral_norm spectral_radius spectral_radius_real --print_config --run_2nd_stage --stop_after_pruning > dump
			python run.py --depth $depth --lr $lr --optimizer adam --batch_size 128 --epochs 400 --history_interval 10 --datasets CIFAR-100 --significance_threshold 1e-4 --compression_modes spectral_norm --l1_penalties 1e-4 --modes_to_compare det det_corr min_eig min_eig_real weight spectral_norm spectral_radius spectral_radius_real --print_config --run_2nd_stage --stop_after_pruning > dump
			python run.py --depth $depth --lr $lr --optimizer adam --batch_size 128 --epochs 400 --history_interval 10 --datasets tiny-imagenet --significance_threshold 1e-4 --compression_modes spectral_norm --l1_penalties 1e-4 --modes_to_compare det det_corr min_eig min_eig_real weight spectral_norm spectral_radius spectral_radius_real --print_config --run_2nd_stage --stop_after_pruning > dump
		done
	done
done
# Static initialization experiments
for i in 1; do
	for depth in 32 56 110; do
		for lr in 1e-3; do #1e-3 for 32 56 110 - 1e-4 for 50
			echo "Exp #"$i" : Training variant = {depth: "$depth", lr: "$lr", static_init}"
			python run.py --depth $depth --lr $lr --optimizer adam --batch_size 128 --epochs 400 --history_interval 10 --datasets CIFAR-10 --significance_threshold 1e-4 --compression_modes spectral_norm --l1_penalties 1e-4 --modes_to_compare det det_corr min_eig min_eig_real weight spectral_norm spectral_radius spectral_radius_real --pretrained_weights "./models/tiny-imagenet/ResNet${depth}_init.h5" --print_config --run_2nd_stage --stop_after_pruning > dump
			python run.py --depth $depth --lr $lr --optimizer adam --batch_size 128 --epochs 400 --history_interval 10 --datasets CIFAR-100 --significance_threshold 1e-4 --compression_modes spectral_norm --l1_penalties 1e-4 --modes_to_compare det det_corr min_eig min_eig_real weight spectral_norm spectral_radius spectral_radius_real --pretrained_weights "./models/CIFAR-10/ResNet${depth}_init.h5" --print_config --run_2nd_stage --stop_after_pruning > dump
			python run.py --depth $depth --lr $lr --optimizer adam --batch_size 128 --epochs 400 --history_interval 10 --datasets tiny-imagenet --significance_threshold 1e-4 --compression_modes spectral_norm --l1_penalties 1e-4 --modes_to_compare det det_corr min_eig min_eig_real weight spectral_norm spectral_radius spectral_radius_real --pretrained_weights "./models/CIFAR-100/ResNet${depth}_init.h5" --print_config --run_2nd_stage --stop_after_pruning > dump
		done
	done
done
# ImageNet pretrained weights initialization experiments
for i in 1; do
	for depth in 50; do
		for lr in 1e-4; do #1e-3 for 32 56 110 - 1e-4 for 50
			echo "Exp #"$i" : Training variant = {depth: "$depth", lr: "$lr", imagenet_init}"
			python run.py --depth $depth --lr $lr --optimizer adam --batch_size 128 --epochs 400 --history_interval 10 --datasets CIFAR-10 --significance_threshold 1e-4 --compression_modes spectral_norm --l1_penalties 1e-4 --modes_to_compare det det_corr min_eig min_eig_real weight spectral_norm spectral_radius spectral_radius_real --pretrained_weights imagenet --print_config --run_2nd_stage --stop_after_pruning > dump
			python run.py --depth $depth --lr $lr --optimizer adam --batch_size 128 --epochs 400 --history_interval 10 --datasets CIFAR-100 --significance_threshold 1e-4 --compression_modes spectral_norm --l1_penalties 1e-4 --modes_to_compare det det_corr min_eig min_eig_real weight spectral_norm spectral_radius spectral_radius_real --pretrained_weights imagenet --print_config --run_2nd_stage --stop_after_pruning > dump
			python run.py --depth $depth --lr $lr --optimizer adam --batch_size 128 --epochs 400 --history_interval 10 --datasets tiny-imagenet --significance_threshold 1e-4 --compression_modes spectral_norm --l1_penalties 1e-4 --modes_to_compare det det_corr min_eig min_eig_real weight spectral_norm spectral_radius spectral_radius_real --pretrained_weights imagenet --print_config --run_2nd_stage --stop_after_pruning > dump
		done
	done
done
echo "Done."
