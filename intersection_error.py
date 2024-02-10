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
    # args.final_epoch = 9

    dataset = args.dataset
    csv_list = args.merged_csv
    
    target_csv = pd.read_csv(csv_list[0])
    
    for csv in csv_list[1:]:
        csv = pd.read_csv(csv)
        target_csv["wrong_1_times"] = target_csv["wrong_1_times"] & csv["wrong_1_times"]
    
    target_csv["our_spurious"] = target_csv["spurious"] & target_csv["wrong_1_times"]
    target_csv["our_nonspurious"] = (target_csv["spurious"] == 0) & target_csv["wrong_1_times"]
    print("Number of our spurious: ", np.sum(target_csv["our_spurious"]))
    print("Number of our nonspurious:", np.sum(target_csv["our_nonspurious"]))
    

    cub_inter_dir = "/nfs/turbo/coe-vvh/jtt/results/CUB/CUB_sample_exp/intersection_error_set"
    if not os.path.exists(cub_inter_dir):
        os.makedirs(cub_inter_dir)
    
    
    spur_precision = np.sum(
            (target_csv[f"wrong_1_times"] == 1) & (target_csv["spurious"] == 1)
        ) / np.sum((target_csv[f"wrong_1_times"] == 1))
    print("Spurious precision", spur_precision)
    spur_recall = np.sum(
        (target_csv[f"wrong_1_times"] == 1) & (target_csv["spurious"] == 1)
    ) / np.sum((target_csv["spurious"] == 1))
    print("Spurious recall", spur_recall)

    if args.dataset == "CUB":
        print("Detailed Error Set Information: ")
        num_00 = np.sum((target_csv['y'] == 0) & (target_csv['place'] == 0)
        & (target_csv["wrong_1_times"]))
        num_01 = np.sum((target_csv['y'] == 0) & (target_csv['place'] == 1)
        & (target_csv["wrong_1_times"]))
        num_10 = np.sum((target_csv['y'] == 1) & (target_csv['place'] == 0)
        & (target_csv["wrong_1_times"]))
        num_11 = np.sum((target_csv['y'] == 1) & (target_csv['place'] == 1)
        & (target_csv["wrong_1_times"]))
        print(f"Waterbird in Water: {num_00}")
        print(f"Waterbird in Land: {num_01}")
        print(f"Landbird in Water: {num_10}")
        print(f"Landbird in Land: {num_11}")



    # print("Detailed Error Set Info:")
    # print(f"")
    # final_epoch = 8
    # if dataset == "MultiNLI":
    #     probs = softmax(np.array(target_csv[[f"pred_prob_None_epoch_{final_epoch}_val_0", f"pred_prob_None_epoch_{final_epoch}_val_1", f"pred_prob_None_epoch_{final_epoch}_val_2"]]), axis = 1)
    #     target_csv["probs_0"] = probs[:,0]
    #     target_csv["probs_1"] = probs[:,1]
    #     target_csv["probs_2"] = probs[:,2]
    #     target_csv["confidence"] = (target_csv['gold_label']==0) * target_csv["probs_0"] + (target_csv['gold_label']==1) * target_csv["probs_1"] + (target_csv['gold_label']==2) * target_csv["probs_2"]
    # else:
    #     probs = softmax(np.array(target_csv[[f"pred_prob_None_epoch_{final_epoch}_val_0", f"pred_prob_None_epoch_{final_epoch}_val_1"]]), axis = 1)
    #     target_csv["probs_0"] = probs[:,0]
    #     target_csv["probs_1"] = probs[:,1]
    #     if dataset == 'CelebA':
    #         target_csv["confidence"] = target_csv["Blond_Hair"] * target_csv["probs_1"] + (1 - target_csv["Blond_Hair"]) * target_csv["probs_0"]
    #     elif dataset == 'CUB':
    #         target_csv["confidence"] = target_csv["y"] * target_csv["probs_1"] + (1 - target_csv["y"]) * target_csv["probs_0"]
        # elif dataset == 'jigsaw':
        #     target_csv["confidence"] = (target_csv["toxicity"] >= 0.5) * target_csv["probs_1"] + (target_csv["toxicity"] < 0.5)  * target_csv["probs_0"]
    
    # target_csv[f"confidence_thres{args.conf_threshold}"] = (target_csv["confidence"] < args.conf_threshold).apply(np.int64)
    if dataset == 'CelebA':
        assert(np.sum(target_csv[f"confidence_thres{args.conf_threshold}"] != target_csv["wrong_1_times"]) == 0)
    
    print(os.path.join(cub_inter_dir, args.save_name))
    target_csv.to_csv(os.path.join(cub_inter_dir, args.save_name))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # merge 
    parser.add_argument("--merged_csv", type=str, default=None, nargs="*")
    # parser.add_argument("--save_path", type=str, default="results/CUB/metadata_csvs")
    parser.add_argument("--save_name", type=str, default="metadata_csv_epoch20_seed12.csv")
    parser.add_argument("--dataset", type=str, default="CUB")

    args = parser.parse_args()
    main(args)
    