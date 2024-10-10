#!/usr/bin/env bash

for data in   'Cora' # 'CoraFull' 'obgnarxiv' 'Reddit' # 'obgnproducts' # 'LastFM' 'Flickr' 'obgnarxiv' 'Cora' 'Citeseer' 'Pubmed' #'Reddit'
do
  for top_node in 100 #100
  do
    for iter_num in 41
    do
    for model in  "GCN" # "APPNP" "GAT" "GraphSAGE" # "GCN"
    do
    for seed in  910 911 912 913 #914 915 916 917 918 919
      do
          # Random Pick
#          python3 -W ignore main_node.py --multiview --random_pick --model $model --dataset $data --seed $seed --iter $iter_num --aug_drop 0.1  --top $top_node --split_by_num
          # Conf + no calib m1
#          python3 -W ignore main_node.py  --conf_pick --model $model --dataset $data --seed $seed --iter $iter_num --aug_drop 0.1  --top $top_node --multiview #--split_by_num
##        python3 -W ignore main_node.py --multiview --conf_pick --model $model --dataset $data --seed $seed --iter $iter_num --aug_drop 0.1  --top $top_node # --FT

      for candidate_num in 200
      do
      if (( top_node < candidate_num - 99)); then
          for sample_num in 500
          do
            for batchPPR in 15
            do
#               # IGP + calib ppr
                python3 -W ignore main_node.py --IGP_pick --model $model --dataset $data --seed $seed --iter $iter_num  --top $top_node --sample_num $sample_num --candidate_num $candidate_num --calib --PageRank --batchPPR $batchPPR  --k_union #--split_by_num
                python3 -W ignore main_node.py --IGP_pick --model $model --dataset $data --seed $seed --iter $iter_num  --top $top_node --sample_num $sample_num --candidate_num $candidate_num --calib  --k_union #--split_by_num
#               python3 -W ignore main_node.py --IGP_pick --model $model --dataset $data --seed $seed --iter $iter_num  --top $top_node --noisy $noisy_portion #--PageRank #--FT
            done
          done
      fi
      done

      done
    done
    done
  done
done

