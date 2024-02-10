from sys import prefix
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
from utils import set_seed, Logger, CSVBatchLogger, Loggers
from data.data import dataset_attributes, shift_types, prepare_data, log_data
from data import dro_dataset
import torch


def merge(csv_dir_list, save_dir, save_name, dataset, logger):
    target_csv = pd.read_csv(csv_dir_list[0])
    for csv in csv_dir_list[1:]:
        csv = pd.read_csv(csv)
        target_csv["wrong_1_times"] = target_csv["wrong_1_times"] & csv["wrong_1_times"]
    
    target_csv["our_spurious"] = target_csv["spurious"] & target_csv["wrong_1_times"]
    target_csv["our_nonspurious"] = (target_csv["spurious"] == 0) & target_csv["wrong_1_times"]
    
    num_sp = int(np.sum(target_csv["our_spurious"]))
    num_nonsp = int(np.sum(target_csv["our_nonspurious"]))
    logger.write("Number of our spurious: ")
    logger.write(str(np.sum(target_csv["our_spurious"])))
    logger.write("\n")
    logger.write("Number of our nonspurious: ")
    logger.write(str(np.sum(target_csv["our_nonspurious"])))
    logger.write("\n")
    
    
    spur_precision = np.sum(
            (target_csv[f"wrong_1_times"] == 1) & (target_csv["spurious"] == 1)
        ) / np.sum((target_csv[f"wrong_1_times"] == 1))
    logger.write("Spurious precision: ")
    logger.write(str(spur_precision))
    logger.write("\n")
    spur_recall = np.sum(
        (target_csv[f"wrong_1_times"] == 1) & (target_csv["spurious"] == 1)
    ) / np.sum((target_csv["spurious"] == 1))
    logger.write("Spurious recall: ")
    logger.write(str(spur_recall))
    logger.write("\n")

    if dataset == "MultiNLI":
        logger.write("\nError Set Detailed Information: \n")
        logger.write("gold_label_random = 0, sentence2_has_negation = 0: ")
        logger.write(str(np.sum((target_csv["wrong_1_times"]) & (target_csv["gold_label"] == 0)
            & (target_csv["sentence2_has_negation"] == 0))))
        logger.write("\n")
        logger.write("gold_label_random = 0, sentence2_has_negation = 1: ")
        logger.write(str(np.sum( (target_csv["wrong_1_times"]) & (target_csv["gold_label"] == 0)
            & (target_csv["sentence2_has_negation"] == 1))))
        logger.write("\n")
        logger.write("gold_label_random = 1, sentence2_has_negation = 0: ")
        logger.write(str(np.sum( (target_csv["wrong_1_times"]) & (target_csv["gold_label"] == 1)
            & (target_csv["sentence2_has_negation"] == 0))))
        logger.write("\n")
        logger.write("gold_label_random = 1, sentence2_has_negation = 1: ")
        logger.write(str(np.sum( (target_csv["wrong_1_times"]) & (target_csv["gold_label"] == 1)
            & (target_csv["sentence2_has_negation"] == 1))))
        logger.write("\n")
        logger.write("gold_label_random = 2, sentence2_has_negation = 0: ")
        logger.write(str(np.sum( (target_csv["wrong_1_times"]) & (target_csv["gold_label"] == 2)
            & (target_csv["sentence2_has_negation"] == 0))))
        logger.write("\n")
        logger.write("gold_label_random = 2, sentence2_has_negation = 1: ")
        logger.write(str(np.sum( (target_csv["wrong_1_times"]) & (target_csv["gold_label"] == 2)
            & (target_csv["sentence2_has_negation"] == 1))))
        logger.write("\n")
    
    target_csv.to_csv(os.path.join(save_dir, save_name))

    return num_sp, num_nonsp
    


def meta_merge(args):
    ### REQUIREMENT ###
    # Directory Name of the Meta_data_csv must be similar, the only difference is epoch_num of the sub-root dir
    # Mode Merge Two is implemented below 
    
    b_input = "withoutb" if args.withoutb else "withb"
    loss_input = args.Losstype 
    seed_list = args.compared_seed
    parent_epoch_list = args.compared_parent_epoch
    dataset = args.dataset
    exp_name = dataset + "_sample_exp"

    suffix_name = "_lr_" + args.lr + "_weight_decay_" + args.weight_decay + "_aux_lambda_" + args.aux_lambda 
    prefix_name = "train_downstream_" + args.method + "_upweight_0_epochs_"
    
    root_dir = os.path.join("results", dataset)
    root_dir = os.path.join(root_dir, exp_name)

    # Save Intersection
    
    inter_name = "intersection_error_set_sgd" if args.sgd else "intersection_error_set"
    if not os.path.exists(os.path.join(root_dir, inter_name)):
        os.makedirs(os.path.join(root_dir, inter_name))
    
    intersection_dir = os.path.join(root_dir, inter_name)
    
    # intersection_log_file_dir = os.path.join(intersection_dir, "log_files")
    # if not os.path.exists(intersection_log_file_dir):
    #     os.makedirs(intersection_log_file_dir)

    seed_names = ""
    for seed in seed_list:
        seed_names += seed
    
    seed_name_in_dir = "seed" + seed_names
    curr_inters_dir = os.path.join(intersection_dir, seed_name_in_dir)

    if not os.path.exists(curr_inters_dir):
        os.makedirs(curr_inters_dir)

    log_file_name = "Merge_Seed_" + seed_names + "_" + b_input + "_total_epoch" + str(args.total_compared_epochs) + ".txt"
    logger = Loggers(os.path.join(curr_inters_dir, log_file_name), "w")

    compared_csv_dir = []
    assert len(parent_epoch_list) == len(seed_list)

    for parent_epoch in parent_epoch_list:

        parent_root_name = prefix_name + parent_epoch + suffix_name
        metadata_aug_files_dir = os.path.join(root_dir, parent_root_name)
        metadata_aug_files_dir = os.path.join(metadata_aug_files_dir, "metadata_aug_files")
        compared_csv_dir.append(metadata_aug_files_dir)
    
    meta_compare_dict = {}
    for idx, seed in enumerate(seed_list):
        meta_compare_dict[seed] = compared_csv_dir[idx]

    # print("We")
    number_of_spurious = []
    number_of_nonspurious = []
    logger.write(f"We are Merging Seed {seed_names}...")
    for epoch in range(int(args.total_compared_epochs)):
        logger.write(f"\n\n\nMerging Seed {seed_names} Epoch {str(epoch)} {b_input}\n")
        csv_name = "metadata_aug_epoch" + str(epoch) + "_" + b_input + "_" + loss_input + ".csv"
        csv_dir_list = []
        for root_csv_dirs in compared_csv_dir:
            csv_dir_list.append(os.path.join(root_csv_dirs, csv_name))
        save_name = "metadata_aug_inter_seed" + seed_names + "_epoch" + str(epoch) + b_input + ".csv"
        curr_sp, curr_non_sp = merge(csv_dir_list, curr_inters_dir, save_name, dataset, logger)
        number_of_spurious.append(curr_sp)
        number_of_nonspurious.append(curr_non_sp)
    # logger.flush()
    logger.close()

    return number_of_spurious, number_of_nonspurious

def merge_wrapper(args):
    if args.combination:
        meta_seed_list = args.compared_seed
        meta_epoch_list = args.compared_parent_epoch
        import numpy as np 
        import itertools
        length = len(meta_seed_list)
        basic_list = np.arange(length)

        all_combinations = list(itertools.combinations(basic_list, args.num_combination))
        # print(meta_seed_list)
        
        meta_list_sp = []
        meta_list_non_sp = []
        average_sp = []
        average_non_sp = []
        for curr_combination in all_combinations:
            args.compared_seed = []
            args.compared_parent_epoch = []
            for idx in curr_combination:
                args.compared_seed.append(meta_seed_list[idx])
                args.compared_parent_epoch.append(meta_epoch_list[idx])
            # print(args.compared_seed)
            list_sp, list_nonsp = meta_merge(args) # return two list of the seed of current combination
            meta_list_sp.append(list_sp) # [ [Seed 1: 0,1,2,3,4], [Seed 2 : 0, 1, 2, 3, 4], [Seed 3: 0, 1, 2,3 ,4 ]] -> [[0, 1, 2, 3, 4]]
            meta_list_non_sp.append(list_nonsp)
        for iterables in zip(*meta_list_sp):
            sum = 0
            num = 0
            for item in iterables:
                sum += item 
                num += 1
            average_sp.append(sum/num)
        for iterables in zip(*meta_list_non_sp):
            sum = 0
            num = 0
            for item in iterables:
                sum += item 
                num += 1
            average_non_sp.append(sum/num)
        
        print("Average Sp and NonSp Information:")
        for idx, (a, b) in enumerate(zip(average_sp, average_non_sp)):
            print(f"Epoch {idx} Average Num-Sp: {a:.2f}")
            print(f"Epoch {idx} Average Num-NonSp: {b:.3f}")
        

    else:
        meta_merge(args)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # merge 
    parser.add_argument("--save_path", type=str, default="results/CUB/metadata_csvs")
    parser.add_argument("--save_name", type=str, default="metadata_csv2345.csv")
    parser.add_argument("--dataset", required=True, type=str, default="jigsaw")
    parser.add_argument("--Losstype", type=str, required=True)
    parser.add_argument("--compared_parent_epoch", type=str, required=True, nargs='*')
    parser.add_argument("--compared_seed", type=str, required=True, nargs='*')
    parser.add_argument("--withoutb", default=False, action="store_true")
    parser.add_argument("--total_compared_epochs", required=True, type=int)
    parser.add_argument("--lr", type=str, default="2e-05")
    parser.add_argument("--weight_decay", type=str, default="0.0")
    parser.add_argument("--aux_lambda", type=str, default="0.5")
    parser.add_argument("--method", default="AUX1", type=str)
    parser.add_argument("--combination", default=False, action="store_true")
    parser.add_argument("--num_combination", type=int, default=2)
    parser.add_argument("--sgd", default=False, action="store_true")

    args = parser.parse_args()
    merge_wrapper(args)
    # meta_merge(args)
    