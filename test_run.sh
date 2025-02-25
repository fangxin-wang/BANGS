#!/usr/bin/env bash

for data in 'Cora'  # 'LastFM' 'Cora' 'Citeseer' 'Pubmed'  # 'Flickr' 'obgnarxiv'
do
  for top_node in  100
  do
    for iter_num in 40
    do
    for model in "GCN" # "GRAND" # "APPNP" "GAT" "GraphSAGE" # "GCN"
    do
    for seed in 910 911 912 #911 912 #913 914 #915 916 917 918 919
      do
        for Train_P in 0.05 0.1 0.2 0.3 0.4
        do
          # Random Pick
#          python3 -W ignore main_node.py --multiview --random_pick --model $model --dataset $data --seed $seed --iter $iter_num --aug_drop 0.1  --top $top_node --split_by_num
          # Conf + no calib m1
#          python3 -W ignore main_node.py  --conf_pick --model $model --dataset $data --seed $seed --iter $iter_num --aug_drop 0.1  --top $top_node --multiview
#          python3 -W ignore main_node.py  --conf_pick --model $model --dataset $data --seed $seed --iter $iter_num  --top $top_node  --calib --train_portion $Train_P

      for candidate_num in 200
      do
      if (( top_node < candidate_num - 99)); then
          for sample_num in 500
          do
            for batchPPR in 10
            do
               # IGP + calib ppr
                python3 -W ignore main_node.py --IGP_pick --model $model --dataset $data --seed $seed --iter $iter_num  --top $top_node --sample_num $sample_num --candidate_num $candidate_num --calib --PageRank --batchPPR $batchPPR --k_union --train_portion $Train_P
#                python3 -W ignore main_node.py --IGP_pick --model $model --dataset $data --seed $seed --iter $iter_num  --top $top_node --sample_num $sample_num --candidate_num $candidate_num  --PageRank --batchPPR $batchPPR  #--k_union
#                python3 -W ignore main_node.py --IGP_pick --model $model --dataset $data --seed $seed --iter $iter_num  --top $top_node --sample_num $sample_num --candidate_num $candidate_num --calib --PageRank --batchPPR $batchPPR
                python3 -W ignore main_node.py --IGP_pick --model $model --dataset $data --seed $seed --iter $iter_num  --top $top_node --sample_num $sample_num --candidate_num $candidate_num --calib --k_union --train_portion $Train_P

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
done

