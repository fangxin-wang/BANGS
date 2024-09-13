#!/usr/bin/env bash

for data in 'Cora' 'Citeseer' # 'LastFM' 'Flicker' 'obgnarxiv' # 'Cora' 'Citeseer' 'Pubmed' #'Reddit'
do
  for top_node in 100
  do
    for iter_num in 40
    do
    for model in "GCN" #"GAT" "GIN" #"GAT" "GraphSAGE" # "GCN"
    do
    for seed in 912 #813 #814 815
      do
      for train_p in 0.05
        do
          for valid_p in 0.15
          do
          for noisy_portion in 0
            do
              # Random Pick
              python3 -W ignore main_node.py --multiview --random_pick --model $model --dataset $data --seed $seed --iter $iter_num --aug_drop 0.1  --top $top_node
              # Conf + no calib m1
              python3 -W ignore main_node.py --multiview --conf_pick --model $model --dataset $data --seed $seed --iter $iter_num --aug_drop 0.1  --top $top_node --train_portion $train_p --valid_portion $valid_p --noisy $noisy_portion
              # IGP + no calib m1
              # python3 -W ignore main_node.py --multiview --IGP_pick --model $model --dataset $data --seed $seed --iter $iter_num  --top $top_node --train_portion $train_p --valid_portion $valid_p --noisy $noisy_portion
              # IGP + calib m0
              python3 -W ignore main_node.py --IGP_pick --model $model --dataset $data --seed $seed --iter $iter_num  --top $top_node --train_portion $train_p --valid_portion $valid_p --noisy $noisy_portion #--PageRank
            done
          done
        done
      done
      done
    done
  done
done



