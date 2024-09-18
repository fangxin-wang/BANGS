#!/usr/bin/env bash

for data in 'obgnarxiv' #'obgnarxiv' # 'obgnproducts' # 'LastFM' 'Flickr' 'obgnarxiv'  # 'Cora' 'Citeseer' 'Pubmed' #'Reddit' 'ogbnmag'
do
  for top_node in 100 #100
  do
    for iter_num in 40
    do
    for model in "GCN" #"GAT" "GraphSAGE" #"GAT" "GraphSAGE" # "GCN"
    do
    for seed in 913 #914ß 915 #910 911 912
      do
      for candidate_num in 200
      do
      if (( top_node < candidate_num - 99)); then
          for noisy_portion in 0
          do
               # Random Pick
#               python3 -W ignore main_node.py --multiview --random_pick --model $model --dataset $data --seed $seed --iter $iter_num --aug_drop 0.1  --top $top_node --FT
#              # Conf + no calib m1
               python3 -W ignore main_node.py --multiview --conf_pick --model $model --dataset $data --seed $seed --iter $iter_num --aug_drop 0.1  --top $top_node --noisy $noisy_portion --FT
               # IGP + calib m0
               python3 -W ignore main_node.py --IGP_pick --model $model --dataset $data --seed $seed --iter $iter_num  --top $top_node --noisy $noisy_portion --FT #--PageRank
          done
      fi
      done
      done
    done
    done
  done
done


#              # IGP + no calib m1
#              python3 -W ignore main_node.py --multiview --IGP_pick --model $model --dataset $data --seed $seed --iter $iter_num  --top $top_node --train_portion $train_p --valid_portion $valid_p --noisy $noisy_portion

