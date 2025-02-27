# # echo "orig - - orig"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/original/test.tsv --model /mnt/data_raid/data/val2017/torch_models-train1667/ --output_dir results-1667-orig-orig
echo "orig - - RH"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/revised_hypothesis/test.tsv --model /mnt/data_raid/data/val2017/torch_models-train1667/ --output_dir results-1667-orig-rh
echo "orig - - RP"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/revised_premise/test.tsv --model /mnt/data_raid/data/val2017/torch_models-train1667/ --output_dir results-1667-orig-rp
echo "orig - - all_revisions"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/revised_combined/test.tsv --model /mnt/data_raid/data/val2017/torch_models-train1667/ --output_dir results-1667-orig-rev_combined
echo "orig - - MNLI"
python3 run.py --do_eval --task nli --dataset multi_nli --model /mnt/data_raid/data/val2017/torch_models-train1667/ --output_dir results-1667-orig-mnli
echo "orig - - fp-rh"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/all_combined/test.tsv --model /mnt/data_raid/data/val2017/torch_models-train1667/ --output_dir results-1667-orig-all_combined 

echo "RP - - orig"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/original/test.tsv --model /mnt/data_raid/data/val2017/torch_models-rp3333/ --output_dir results-1667-rp-orig
echo "RP - - RH"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/revised_hypothesis/test.tsv --model /mnt/data_raid/data/val2017/torch_models-rp3333/ --output_dir results-1667-rp-rh
echo "RP - - RP"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/revised_premise/test.tsv --model /mnt/data_raid/data/val2017/torch_models-rp3333/ --output_dir results-1667-rp-rp
echo "RP - - all_revisions"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/revised_combined/test.tsv --model /mnt/data_raid/data/val2017/torch_models-rp3333/ --output_dir results-1667-rev-all_combined 
echo "RP - - MNLI"
python3 run.py --do_eval --task nli --dataset multi_nli --model /mnt/data_raid/data/val2017/torch_models-rp3333/ --output_dir results-1667-rp-mnli
echo "orig - - fp-rh"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/all_combined/test.tsv --model /mnt/data_raid/data/val2017/torch_models-rp3333/ --output_dir results-1667-rp-all_combined

echo "RH - - orig"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/original/test.tsv --model /mnt/data_raid/data/val2017/torch_models-rh3333/ --output_dir results-1667-rh-orig
echo "RH - - RH"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/revised_hypothesis/test.tsv --model /mnt/data_raid/data/val2017/torch_models-rh3333/ --output_dir results-1667-rh-rh
echo "RH - - RP"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/revised_premise/test.tsv --model /mnt/data_raid/data/val2017/torch_models-rh3333/ --output_dir results-1667-rh-rp
echo "RH - - all_revisions"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/revised_combined/test.tsv --model /mnt/data_raid/data/val2017/torch_models-rh3333/ --output_dir results-1667-rh-all_combined
echo "RH - - MNLI"
python3 run.py --do_eval --task nli --dataset multi_nli --model /mnt/data_raid/data/val2017/torch_models-rh3333/ --output_dir results-1667-rh-mnli
echo "RH - - fp-rh"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/all_combined/test.tsv --model /mnt/data_raid/data/val2017/torch_models-rh3333/ --output_dir results-1667-rh-rev-all-combined

echo "all_revisions - - orig"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/original/test.tsv --model /mnt/data_raid/data/val2017/torch_models-all_comb-8331/ --output_dir results-1667-all-orig
echo "all_revisions - - RH"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/revised_hypothesis/test.tsv --model /mnt/data_raid/data/val2017/torch_models-all_comb-8331/ --output_dir results-1667-all-rh
echo "all_revisions - - RP"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/revised_premise/test.tsv --model /mnt/data_raid/data/val2017/torch_models-all_comb-8331/ --output_dir results-1667-all-rp
echo "all_revisions - - all_revisions"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/revised_combined/test.tsv --model /mnt/data_raid/data/val2017/torch_models-all_comb-8331/ --output_dir  results-1667-rh-rev-all-combined
echo "all_revisions - - MNLI"
python3 run.py --do_eval --task nli --dataset multi_nli --model /mnt/data_raid/data/val2017/torch_models-all_comb-8331/ --output_dir results-1667-all_combined-mnli
echo "all_revisions - - fp-rh"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/all_combined/test.tsv --model /mnt/data_raid/data/val2017/torch_models-rh3333/ --output_dir results-1667-all-all_combined 

echo "rp-rh- - orig"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/original/test.tsv --model /mnt/data_raid/data/val2017/torch_models-rhrp --output_dir results-1667-rprh-orig
echo "rp-rh - - RH"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/revised_hypothesis/test.tsv --model /mnt/data_raid/data/val2017/torch_models-rhrp --output_dir results-1667-rprh-rh
echo "rp-rh- - RP"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/revised_premise/test.tsv --model /mnt/data_raid/data/val2017/torch_models-rhrp --output_dir results-1667-rprh-rp
echo "rp-rh- - all_revisions"
python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/revised_combined/test.tsv --model /mnt/data_raid/data/val2017/torch_models-rhrp --output_dir  results-1667-rprh-rev-all-combined
echo "rp-rh - - MNLI"
python3 run.py --do_eval --task nli --dataset multi_nli --model /mnt/data_raid/data/val2017/torch_models-rhrp --output_dir results-1667-rprh-mnli
# echo "rp-rh - - fp-rh"
# python3 run.py --do_eval --task nli --dataset ../counterfactually-augmented-data/NLI/all_combined/test.tsv --model /mnt/data_raid/data/val2017/torch_models-rhrp --output_dir results-1667-rprh-all_combined 
# echo "MNLI -  - ORIG"
# echo "MNLI -  - test"
# python3 run.py --do_eval --task nli --dataset MultiNLI --model /mnt/data_raid/data/val2017/torch_models-train1667  --output_dir results-orig-mli-final/
# echo "MNLI - orig - RP"
# echo "MNLI - orig - RH"
# echo "MNLI - orig - all_revisions"
