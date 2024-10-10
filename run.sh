#!/usr/bin/env bash

for model in "GCN" #"GIN" "GraphSAGE" "GAT"
do
  for data in  "Flickr" # "Citeseer" "Pubmed" "LastFM" "APh" "Cora"
  do
    for seed in  912 #1111 #1234 #4321 42 4399 128
    do
      #python3 -W ignore baseline/m3s.py --multiview --dataset $data --model $model --iter 5 --seed $seed
      python3 -W ignore baseline/drgst.py --dataset $data --model $model --seed $seed
    done
  done
done

#for model in "GCN"
#do
#  for seed in 1111
#  do
#    python3 -W ignore main_node.py  --dataset Cora --model $model --top 100 --iter 25 --aug_drop 0.1 --seed $seed --conf_pick --multiview
#    python3 -W ignore main_node.py --dataset Citeseer --model $model --top 70 --iter 30  --aug_drop 0.05 --seed $seed --conf_pick  --multiview
#    python3 -W ignore main_node.py  --dataset Pubmed --model $model --top 1000 --iter 20 --aug_drop 0.1 --seed $seed --conf_pick  --multiview
#    python3 -W ignore main_node.py  --dataset LastFM --model $model --top 400 --iter 20 --aug_drop 0.2 --seed $seed --conf_pick --train_portion 0.05 --valid_portion 0.15  --multiview
#    #python3 -W ignore main_node.py  --dataset APh --model $model --top 200 --iter 35 --aug_drop 0.1 --seed $seed --random_pick --train_portion 0.05 --valid_portion 0.15  --multiview
#  done
#done




