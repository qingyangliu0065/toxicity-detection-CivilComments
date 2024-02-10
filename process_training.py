import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from scipy.special import softmax
import argparse
import subprocess
from train import run_epoch
from loss import LossComputer
from data.folds import Subset, ConcatDataset
from utils import set_seed, Logger, CSVBatchLogger, log_args, get_model, hinge_loss
from data.data import dataset_attributes, shift_types, prepare_data, log_data
from data import dro_dataset
import torch



def main(args):
    if args.final_epoch is None:
        if args.use_weighted_spurious_score:
            with open(os.path.join(args.log_dir_root, args.folder_name, 'model_outputs/best_test_weighted_epoch.txt'), 'r') as file:
                args.final_epoch = int(float(file.read()))
        else:
            with open(os.path.join(args.log_dir_root, args.folder_name, 'model_outputs/best_test_epoch.txt'), 'r') as file:
                args.final_epoch = int(float(file.read()))

    best_epoch_info = args.final_epoch
    info_dict = {}
    with open(os.path.join(args.log_dir_root, args.folder_name, 'model_outputs/information.txt'), 'r') as f:
        for line in f:
            (k, v) = line.split()
            info_dict[k] = v
    print(info_dict)
   
    # args.final_epoch = 9
    final_epoch = args.final_epoch if not args.ProcessWhole else 0 

    num_processes = info_dict["n_epochs"] if args.ProcessWhole else 1
    num_processes = args.total_epochs_evaluate if args.total_epochs_evaluate is not None else num_processes 
    
    if args.loadModel is not None:
        final_epoch = args.loadModel

    dataset = args.dataset
    # CHANGE THESE FOLDERS
    exp_name = args.exp_name
    folder_name = args.folder_name
    data_dir = f"results/{args.dataset}/{exp_name}/{folder_name}/model_outputs/"
    if args.dataset == 'CelebA':
        metadata_path = "./celebA/data/metadata.csv"
    elif args.dataset == 'MultiNLI':
        metadata_path = "./multinli/data/metadata.csv"
    elif args.dataset == 'CUB':
        metadata_path = "./cub/data/waterbird_complete95_forest2water2/metadata.csv"
    elif args.dataset == "jigsaw":
        metadata_path = "./jigsaw/data/all_data_with_identities.csv"
    else: 
        raise NotImplementedError 
    
    for final_epoch in range(int(num_processes)):
        args.loadModel = final_epoch if args.ProcessWhole else None
        if not args.ProcessWhole:
            final_epoch = best_epoch_info if args.loadModel is None else args.loadModel
        # Load in train df and wrong points, this is the main part
        train_df = pd.read_csv(os.path.join(data_dir, f"output_train_epoch_{final_epoch}.csv"))
        train_df = train_df.sort_values(f"indices_None_epoch_{final_epoch}_val")
        train_df["wrong_1_times"] = (1.0 * (train_df[f"y_pred_None_epoch_{final_epoch}_val"] != train_df[f"y_true_None_epoch_{final_epoch}_val"])).apply(np.int64)
        print("Total wrong", np.sum(train_df['wrong_1_times']), "Total points", len(train_df))
        # print("first len: ", len(train_df[f"indices_None_epoch_{final_epoch}_val"]))
        # Merge with original features (could be optional)
        original_df = pd.read_csv(metadata_path)
        # if dataset == "CUB":
        #     original_train_df = original_df[original_df["split"] == 0]
        
        if dataset == "jigsaw":
            original_train_df = original_df[original_df["split"] == "train"]
        else:
            original_train_df = original_df[original_df["split"] == 0]

        if dataset == "jigsaw" or dataset == "MultiNLI":
        # if dataset == "CelebA" or dataset == "jigsaw" or dataset == "MultiNLI":
            original_train_df = original_train_df.drop(['Unnamed: 0'], axis=1)
            # print("first len: ", len(train_df[f"indices_None_epoch_{final_epoch}_val"]))
            # train_df[f"indices_None_epoch_{final_epoch}_val"] = train_df["indices_None_epoch_{final_epoch}_val"].str.strip()

        merged_csv = original_train_df.join(train_df.set_index(f"indices_None_epoch_{final_epoch}_val"))
        
        if dataset == "CUB":
            merged_csv["spurious"] = merged_csv['y'] != merged_csv["place"]
        elif dataset == "CelebA":
            merged_csv = merged_csv.replace(-1, 0)
            assert 0 == np.sum(merged_csv[merged_csv["split"] == 0]["Blond_Hair"] != merged_csv[merged_csv["split"] == 0][f"y_true_None_epoch_{final_epoch}_val"])
            merged_csv["spurious"] = (merged_csv["Blond_Hair"] == merged_csv["Male"]) 
        elif dataset == "jigsaw":
            merged_csv["spurious"] = original_train_df["toxicity"] >= 0.5
            print("merged_csv len of toxicity: ", len(merged_csv["toxicity"]))
        elif dataset == "MultiNLI":
            # merged_csv["spurious"] = (
            #         (merged_csv["gold_label"] == 0)
            #         & (merged_csv["sentence2_has_negation"] == 0)
            #     ) | (
            #         (merged_csv["gold_label"] == 1)
            #         & (merged_csv["sentence2_has_negation"] == 1)
            #     )
            merged_csv["spurious"] = (
                    (merged_csv["gold_label"] == 2)
                    & (merged_csv["sentence2_has_negation"] == 1)
                ) | (
                    (merged_csv["gold_label"] == 1)
                    & (merged_csv["sentence2_has_negation"] == 1)
                )
        else: 
            raise NotImplementedError
        
        print("Number of spurious", np.sum(merged_csv['spurious']))
        
        # Make columns for our spurious and our nonspurious
        merged_csv["our_spurious"] = merged_csv["spurious"] & merged_csv["wrong_1_times"]
        merged_csv["our_nonspurious"] = (merged_csv["spurious"] == 0) & merged_csv["wrong_1_times"]
        print("Number of our spurious: ", np.sum(merged_csv["our_spurious"]))
        print("Number of our nonspurious:", np.sum(merged_csv["our_nonspurious"]))
        
        if dataset == "MultiNLI":
            print("\nDetailed Error Set Information: ")
            print("gold_label_random = 0, sentence2_has_negation = 0: ", 
            np.sum( (merged_csv["wrong_1_times"]) & (merged_csv["gold_label"] == 0)
            & (merged_csv["sentence2_has_negation"] == 0)))

            print("gold_label_random = 0, sentence2_has_negation = 1: ", 
            np.sum( (merged_csv["wrong_1_times"]) & (merged_csv["gold_label"] == 0)
            & (merged_csv["sentence2_has_negation"] == 1)))

            print("gold_label_random = 1, sentence2_has_negation = 0: ", 
            np.sum( (merged_csv["wrong_1_times"]) & (merged_csv["gold_label"] == 1)
            & (merged_csv["sentence2_has_negation"] == 0)))

            print("gold_label_random = 1, sentence2_has_negation = 1: ", 
            np.sum( (merged_csv["wrong_1_times"]) & (merged_csv["gold_label"] == 1)
            & (merged_csv["sentence2_has_negation"] == 1)))

            print("gold_label_random = 2, sentence2_has_negation = 0: ", 
            np.sum( (merged_csv["wrong_1_times"]) & (merged_csv["gold_label"] == 2)
            & (merged_csv["sentence2_has_negation"] == 0)))

            print("gold_label_random = 2, sentence2_has_negation = 1: ", 
            np.sum( (merged_csv["wrong_1_times"]) & (merged_csv["gold_label"] == 2)
            & (merged_csv["sentence2_has_negation"] == 1)))
        elif dataset == "CUB":
            print("Detailed Error Set Information: ")
            num_00 = np.sum((merged_csv['y'] == 0) & (merged_csv['place'] == 0)
            & (merged_csv["wrong_1_times"]))
            num_01 = np.sum((merged_csv['y'] == 0) & (merged_csv['place'] == 1)
            & (merged_csv["wrong_1_times"]))
            num_10 = np.sum((merged_csv['y'] == 1) & (merged_csv['place'] == 0)
            & (merged_csv["wrong_1_times"]))
            num_11 = np.sum((merged_csv['y'] == 1) & (merged_csv['place'] == 1)
            & (merged_csv["wrong_1_times"]))
            print(f"Waterbird in Water: {num_00}")
            print(f"Waterbird in Land: {num_01}")
            print(f"Landbird in Water: {num_10}")
            print(f"Landbird in Land: {num_11}")
        elif dataset == "jigsaw":
            print("\nDetailed Error Set Information: ")
            num_00 = np.sum((merged_csv['identity_any'] == 0) & (merged_csv["toxicity"] < 0.5)
            & (merged_csv["wrong_1_times"]))
            num_01 = np.sum((merged_csv['identity_any'] == 1) & (merged_csv["toxicity"] < 0.5)
            & (merged_csv["wrong_1_times"]))
            num_10 = np.sum((merged_csv['identity_any'] == 0) & (merged_csv["toxicity"] >= 0.5)
            & (merged_csv["wrong_1_times"]))
            num_11 = np.sum((merged_csv['identity_any'] == 1) & (merged_csv["toxicity"] >= 0.5)
            & (merged_csv["wrong_1_times"]))
            print(f"Not-toxic No identity: {num_00}")
            print(f"Not-toxic identity: {num_01}")
            print(f"Toxic No identity: {num_10}")
            print(f"Not-toxic Identity: {num_11}")
        elif dataset == "CelebA":
            print("\nDetailed Error Set Information: ")
            num_00 = np.sum((merged_csv['Blond_Hair'] == 0) & (merged_csv["Male"] < 0.5)
            & (merged_csv["wrong_1_times"]))
            num_01 = np.sum((merged_csv['Blond_Hair'] == 1) & (merged_csv["Male"] < 0.5)
            & (merged_csv["wrong_1_times"]))
            num_10 = np.sum((merged_csv['Blond_Hair'] == 0) & (merged_csv["Male"] >= 0.5)
            & (merged_csv["wrong_1_times"]))
            num_11 = np.sum((merged_csv['Blond_Hair'] == 1) & (merged_csv["Male"] >= 0.5)
            & (merged_csv["wrong_1_times"]))
            print(f"Not-blond Female: {num_00}")
            print(f"Blond Female: {num_01}")
            print(f"Not-blond Male: {num_10}")
            print(f"Blond Male: {num_11}")



        train_probs_df= merged_csv.fillna(0)
        
        # Output spurious recall and precision
        spur_precision = np.sum(
                (merged_csv[f"wrong_1_times"] == 1) & (merged_csv["spurious"] == 1)
            ) / np.sum((merged_csv[f"wrong_1_times"] == 1))
        print("Spurious precision", spur_precision)
        spur_recall = np.sum(
            (merged_csv[f"wrong_1_times"] == 1) & (merged_csv["spurious"] == 1)
        ) / np.sum((merged_csv["spurious"] == 1))
        print("Spurious recall", spur_recall)
        
        # Find confidence (just in case doing threshold)
        if dataset == "MultiNLI":
            probs = softmax(np.array(train_probs_df[[f"pred_prob_None_epoch_{final_epoch}_val_0", f"pred_prob_None_epoch_{final_epoch}_val_1", f"pred_prob_None_epoch_{final_epoch}_val_2"]]), axis = 1)
            train_probs_df["probs_0"] = probs[:,0]
            train_probs_df["probs_1"] = probs[:,1]
            train_probs_df["probs_2"] = probs[:,2]
            train_probs_df["confidence"] = (train_probs_df['gold_label']==0) * train_probs_df["probs_0"] + (train_probs_df['gold_label']==1) * train_probs_df["probs_1"] + (train_probs_df['gold_label']==2) * train_probs_df["probs_2"]
        else:
            probs = softmax(np.array(train_probs_df[[f"pred_prob_None_epoch_{final_epoch}_val_0", f"pred_prob_None_epoch_{final_epoch}_val_1"]]), axis = 1)
            train_probs_df["probs_0"] = probs[:,0]
            train_probs_df["probs_1"] = probs[:,1]
            if dataset == 'CelebA':
                train_probs_df["confidence"] = train_probs_df["Blond_Hair"] * train_probs_df["probs_1"] + (1 - train_probs_df["Blond_Hair"]) * train_probs_df["probs_0"]
            elif dataset == 'CUB':
                train_probs_df["confidence"] = train_probs_df["y"] * train_probs_df["probs_1"] + (1 - train_probs_df["y"]) * train_probs_df["probs_0"]
            elif dataset == 'jigsaw':
                train_probs_df["confidence"] = (train_probs_df["toxicity"] >= 0.5) * train_probs_df["probs_1"] + (train_probs_df["toxicity"] < 0.5)  * train_probs_df["probs_0"]
        
        train_probs_df[f"confidence_thres{args.conf_threshold}"] = (train_probs_df["confidence"] < args.conf_threshold).apply(np.int64)
        if dataset == 'CelebA':
            assert(np.sum(train_probs_df[f"confidence_thres{args.conf_threshold}"] != train_probs_df["wrong_1_times"]) == 0)
        
        # Save csv into new dir for the run, and generate downstream runs
        if not os.path.exists(f"results/{dataset}/{exp_name}/train_downstream_{folder_name}/final_epoch{final_epoch}"):
            os.makedirs(f"results/{dataset}/{exp_name}/train_downstream_{folder_name}/final_epoch{final_epoch}")
        root = f"results/{dataset}/{exp_name}/train_downstream_{folder_name}/final_epoch{final_epoch}"
        train_probs_df.to_csv(f"{root}/metadata_aug.csv")

        # Process Whole
        # if args.ProcessWhole:
        meta_aug_csv_root = f"results/{dataset}/{exp_name}/train_downstream_{folder_name}"
        if not os.path.exists(f"{root}/metadata_aug_files"):
            os.makedirs(f"{root}/metadata_aug_files")
        if not os.path.exists(f"{meta_aug_csv_root}/metadata_aug_files"):
            os.makedirs(f"{meta_aug_csv_root}/metadata_aug_files")
        loss_type = info_dict["loss_type"]
        b_input = "withoutb" if args.withoutb else "withb"
        train_probs_df.to_csv(f"{meta_aug_csv_root}/metadata_aug_files/metadata_aug_epoch{final_epoch}_{b_input}_{loss_type}.csv")
        train_probs_df.to_csv(f"{root}/metadata_aug_files/metadata_aug_epoch{final_epoch}_{b_input}_{loss_type}.csv")
        root = f"{exp_name}/train_downstream_{folder_name}/final_epoch{final_epoch}"
        print(f"Epoch {final_epoch} save!")
        print("\n"*3)

        
        root = f"{exp_name}/train_downstream_{folder_name}/final_epoch{final_epoch}"
        print(str(os.path.join(args.log_dir_root, args.folder_name)))
        sbatch_command = (
                f"python generate_downstream.py --exp_name {root} --lr {args.lr} --n_epochs {args.n_epochs} --weight_decay {args.weight_decay} --method JTT --dataset {args.dataset} --aug_col {args.aug_col} --log_dir_old {str(os.path.join(args.log_dir_root, args.folder_name))}" 
                + (f" --batch_size {args.batch_size}" if args.batch_size else "")
                + (f" --method {args.method}" if args.method else "")
                + (f" --use_weighted_spurious_score" if args.use_weighted_spurious_score else "")
                + (f" --use_confidence" if args.use_confidence else "")
                + (f" --confidence {args.confidence}" if args.use_confidence else "")
                + (f" --subsample_propotion {args.subsample_propotion}" if args.subsample_propotion else "")
                + (f" --metadata_path {args.metadata_path}" if args.metadata_path is not None else "")
                + (f" --loss {args.loss}")
                + (f" --loadModel {args.loadModel}" if args.loadModel is not None else "")
                + (f" --set_seed {args.set_seed}" if args.set_seed is not None else "")
                + (f" --load_new_model" if args.load_new_model else "")
            )
        print(sbatch_command)
        # if args.deploy:
        #     subprocess.run(sbatch_command, check=True, shell=True)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="CelebA", help="CUB, CelebA, or MultiNLI"
    )
    parser.add_argument(
        "--final_epoch",
        type=int,
        default=None,
        help="last epoch in training",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--deploy", action="store_true", default=False)
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--aug_col", type=str, default='wrong_1_times')
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--folder_name", type=str, required=True)
    parser.add_argument("--log_dir_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)

    # AUX 2
    parser.add_argument("--use_weighted_spurious_score", default=False, action="store_true")
    parser.add_argument("--use_confidence", default=False, action="store_true")
    parser.add_argument("--confidence", default=0.5, type=float)
    parser.add_argument("--subsample_propotion", default=0.0, type=float)
    parser.add_argument("--metadata_path", type=str, default=None)
    
    # Process Training
    parser.add_argument("--old_epoch", default=0, type=int)
    parser.add_argument("--old_lr", default=1e-5, type=float)
    parser.add_argument("--old_lambda", default=0.5, type=float)

    # Loss
    parser.add_argument("--loss", type=str, default="CrossEntropy", 
    help="GCE, CrossEntropy, Sqaured Loss, LabelSmoothing, LabelSmoothingSquaredLoss")
    parser.add_argument("--loadModel", type=int, default=None)
    parser.add_argument("--ProcessWhole", action="store_true", default=False)
    parser.add_argument("--total_epochs_evaluate", type=int, default=None)
    parser.add_argument("--withoutb", default=False, action="store_true")
    parser.add_argument("--set_seed", default=None, type=int)
    parser.add_argument("--load_new_model", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
    