echo "orig - - MNLI"
python3 run.py --do_eval --task nli --dataset multi_nli --and_if_false_not_true --model /mnt/data_raid/data/val2017/torch_models-sub0 --output_dir results-8k-sub0
python3 run.py --do_eval --task nli --dataset multi_nli --and_if_false_not_true --model /mnt/data_raid/data/val2017/torch_models-sub1 --output_dir results-8k-sub1
python3 run.py --do_eval --task nli --dataset multi_nli --and_if_false_not_true --model /mnt/data_raid/data/val2017/torch_models-sub2 --output_dir results-8k-sub2
python3 run.py --do_eval --task nli --dataset multi_nli --and_if_false_not_true --model /mnt/data_raid/data/val2017/torch_models-sub3 --output_dir results-8k-sub3
python3 run.py --do_eval --task nli --dataset multi_nli --and_if_false_not_true --model /mnt/data_raid/data/val2017/torch_models-sub4 --output_dir results-8k-sub4

# artificat to snlp
# python3 run.py --do_eval --task nli --dataset multi_nli --and_if_false_not_true --model /mnt/data_raid/data/val2017/torch_models-train1667/ --output_dir results-1667-if
# python3 run.py --do_eval --task nli --dataset multi_nli --and_if_false_not_true --model /mnt/data_raid/data/val2017/torch_models-rp3333/ --output_dir results-1667-rp-if
# python3 run.py --do_eval --task nli --dataset multi_nli --and_if_false_not_true --model /mnt/data_raid/data/val2017/torch_models-rh3333/ --output_dir results-1667-rh-if
# python3 run.py --do_eval --task nli --dataset multi_nli --and_if_false_not_true --model /mnt/data_raid/data/val2017/torch_models-rhrp --output_dir results-1667-rprh-if