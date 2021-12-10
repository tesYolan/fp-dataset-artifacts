# this script is for getting original data to the Learing the Difference that Makes a Difference with Counterfactually-augmented data
# --task nli --dataset counterfactually-augmented-data/NLI/all_combined/test.tsv 
echo "orig - - orig"
echo "orig - - MNLI"
echo "orig - - test"
echo "orig - - RP"
echo "orig - - RH"
echo "orig - - all_revisions"

echo "MNLI -  - ORIG"
echo "MNLI -  - test"
echo "MNLI - orig - RP"
echo "MNLI - orig - RH"
echo "MNLI - orig - all_revisions"

echo "RP-  - ORIG"
echo "RP -  - MLI"
echo "RP - orig - RP"
echo "RP - orig - RH"
echo "RP - orig - all_revisions"

echo "RH-  - ORIG"
echo "RH -  - MLI"
echo "RH - orig - RP"
echo "RH - orig - RH"
echo "RH - orig - all_revisions"

echo "all_combined-  - ORIG"
echo "all_combined -  - MLI"
echo "all_combined - orig - RP"
echo "all_combined - orig - RH"
echo "all_combined - orig - all_revisions"


# python3 run.py --do_train --max_length 50 --task nli --dataset ../counterfactually-augmented-data/NLI/original/train.tsv --output_dir /mnt/data_raid/data/val2017/torch_models-train1667 --num_train_epochs 20 --per_device_train_batch_size 16
# echo "Just train with 1667 samples taken from the revised premise"
# python3 run.py --do_train --max_length 50 --task nli --dataset ../counterfactually-augmented-data/NLI/revised_premise/train.tsv --output_dir /mnt/data_raid/data/val2017/torch_models-rp3333 --num_train_epochs 20 --per_device_train_batch_size 16
# echo "Just train with 1667 samples taken from the revised hypothesis"
python3 run.py --do_train --max_length 50 --task nli --dataset ../counterfactually-augmented-data/NLI/revised_hypothesis/train.tsv --output_dir /mnt/data_raid/data/val2017/torch_models-rh3333 --num_train_epochs 20 --per_device_train_batch_size 16
# echo "Just train with 1667 samples taken from the all_comb"
# python3 run.py --do_train --max_length 50 --task nli --dataset ../counterfactually-augmented-data/NLI/all_combined/train.tsv --output_dir /mnt/data_raid/data/val2017/torch_models-all_comb-8331 --num_train_epochs 20 --per_device_train_batch_size 16
echo "Just train with 1667 samples taken from the revised hypothesis - revised premise"
python3 run.py --do_train --max_length 50 --task nli --dataset ../counterfactually-augmented-data/NLI/revised_combined/train.tsv --output_dir /mnt/data_raid/data/val2017/torch_models-rhrp --num_train_epochs 20 --per_device_train_batch_size 16