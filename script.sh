# this script is for getting original data to the Learing the Difference that Makes a Difference with Counterfactually-augmented data
# --task nli --dataset counterfactually-augmented-data/NLI/all_combined/test.tsv 
echo "Just train with 1667 samples taken from the original data"
python3 run.py --do_train --max_length 50 --task nli --dataset ../counterfactually-augmented-data/NLI/original/train.tsv --output_dir /mnt/data_raid/data/val2017/torch_models-train1667 --num_train_epochs 20 --per_device_train_batch_size 16
echo "Just train with 1667 samples taken from the revised premise"
python3 run.py --do_train --max_length 50 --task nli --dataset ../counterfactually-augmented-data/NLI/revised_premise/train.tsv --output_dir /mnt/data_raid/data/val2017/torch_models-rp3333 --num_train_epochs 20 --per_device_train_batch_size 16
echo "Just train with 1667 samples taken from the revised hypothesis"
python3 run.py --do_train --max_length 50 --task nli --dataset ../counterfactually-augmented-data/NLI/revised_hypothesis/train.tsv --output_dir /mnt/data_raid/data/val2017/torch_models-rh3333 --num_train_epochs 20 --per_device_train_batch_size 16
echo "Just train with 1667 samples taken from the all_comb"
python3 run.py --do_train --max_length 50 --task nli --dataset ../counterfactually-augmented-data/NLI/all_combined/train.tsv --output_dir /mnt/data_raid/data/val2017/torch_models-all_comb-8331 --num_train_epochs 20 --per_device_train_batch_size 16