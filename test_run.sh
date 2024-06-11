#!/usr/bin/env bash

for data in  'Pubmed' 'Citeseer' 'Cora' 'LastFM'
do
  for top_node in 100
  do
    for iter_num in 40
    do
    for model in "GCN" # "GIN" "GAT" "GraphSAGE"
    do
    for seed in  1204 1234 6666 8888 1111
      do
      for train_p in 0.05
        do
          for valid_p in 0.15
          do
          for noisy_portion in 0
            do
#              # Conf + no calib m1
#              python3 -W ignore main_node.py --multiview --conf_pick --model $model --dataset $data --seed $seed --iter $iter_num --aug_drop 0.1  --top $top_node --train_portion $train_p --valid_portion $valid_p
#              # IGP + no calib m1
#              python3 -W ignore main_node.py --multiview --IGP_pick --model $model --dataset $data --seed $seed --iter $iter_num  --top $top_node --train_portion $train_p --valid_portion $valid_p --noisy $noisy_portion
              # IGP + calib m0
              python3 -W ignore main_node.py --IGP_pick --model $model --dataset $data --seed $seed --iter $iter_num  --top $top_node --train_portion $train_p --valid_portion $valid_p --noisy $noisy_portion
            done
          done
        done
      done
      done
    done
  done
done


