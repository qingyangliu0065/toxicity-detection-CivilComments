import os 
import pandas as pd 
import numpy as np 
import torch 
from train import run_epoch
from data import dro_dataset
import argparse
from loss import LossComputer

def divide_data(args):
    metadata_path = "./jigsaw/data/all_data_with_identities.csv"
    data = pd.read_csv(metadata_path)
    data = data[data["split"]=="test"]
    data['num_words'] = data['comment_text'].str.split().str.len()

    short_data = data[data['num_words'] <= args.short_text]
    long_data = data[data['num_words'] >= args.long_text] 


    model = torch.load(args.model_path)

    small_loader = dro_dataset.get_loader(short_data,
                                        train=False,
                                        reweight_groups=None)
    long_loader = dro_dataset.get_loader(long_data,
                                        train=False,
                                        reweight_groups=None)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    small_test_loss_computer = LossComputer(
                criterion,
                loss_type='erm',
                dataset=short_data,
                dataset_name='jigsaw',
                step_size=args.robust_step_size,
                alpha=args.alpha,
                gamma=args.gamma,
                normalize_loss=args.use_normalized_loss,
                btl=args.btl,
                min_var_weight=args.minimum_variational_weight,
                joint_dro_alpha=args.joint_dro_alpha,
            )
    long_test_loss_computer = LossComputer(
                criterion,
                loss_type='erm',
                dataset=long_data,
                dataset_name='jigsaw',
                step_size=args.robust_step_size,
                alpha=args.alpha,
                gamma=args.gamma,
                normalize_loss=args.use_normalized_loss,
                btl=args.btl,
                min_var_weight=args.minimum_variational_weight,
                joint_dro_alpha=args.joint_dro_alpha,
            )
    print("###################### RESULT FOR SMALL LOADER #########################")
    run_epoch(epoch=0, model=model, loader=small_loader, optimizer=None, loss_computer=small_test_loss_computer, logger=None, 
              csv_logger=None, args=None, is_training=False)
    
    print("###################### RESULT FOR BIG LOADER #########################")
    run_epoch(epoch=0, model=model, loader=long_loader,optimizer=None, loss_computer=long_test_loss_computer, logger=None, 
              csv_logger=None, args=None, is_training=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Settin
                        
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--project_name", type=str, default="spurious", help="wandb project name")
    # Confounders
    parser.add_argument("-t", "--target_name")
    parser.add_argument("-c", "--confounder_names", nargs="+")
    parser.add_argument("--up_weight", type=int, default=0)
    # Resume?
    parser.add_argument("--resume", default=False, action="store_true")
    # Label shifts
    parser.add_argument("--minority_fraction", type=float)
    parser.add_argument("--imbalance_ratio", type=float)
    # Data
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--reweight_groups", action="store_true",
                        default=False,
                        help="set to True if loss_type is group DRO")
    parser.add_argument("--augment_data", action="store_true", default=False)
    parser.add_argument("--val_fraction", type=float, default=0.1)

    # Objective
    parser.add_argument("--loss_type", default="erm",
                        choices=["erm", "group_dro", "joint_dro"])
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--generalization_adjustment", default="0.0")
    parser.add_argument("--automatic_adjustment",
                        default=False,
                        action="store_true")
    parser.add_argument("--robust_step_size", default=0.01, type=float)
    parser.add_argument("--joint_dro_alpha", default=1, type=float,
                         help=("Size param for CVaR joint DRO."
                               " Only used if loss_type is joint_dro"))
    parser.add_argument("--use_normalized_loss",
                        default=False,
                        action="store_true")
    parser.add_argument("--btl", default=False, action="store_true")
    parser.add_argument("--hinge", default=False, action="store_true")
    
    parser.add_argument("--train_from_scratch",
                        action="store_true",
                        default=False)
    parser.add_argument('--aux_lambda', type=float, default=None)
    # Method
    parser.add_argument("--method", type=str, default="JTT")
    # Optimization
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--minimum_variational_weight", type=float, default=0)
    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show_progress", default=False, action="store_true")
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--log_dir_old", type=str, default=None)
    parser.add_argument("--log_every", default=50, type=int)
    parser.add_argument("--save_step", type=int, default=10)
    parser.add_argument("--save_best", action="store_true", default=False)
    parser.add_argument("--save_last", action="store_true", default=False)
    parser.add_argument("--use_bert_params", type=int, default=1)
    parser.add_argument("--num_folds_per_sweep", type=int, default=5)
    parser.add_argument("--num_sweeps", type=int, default=4)
    parser.add_argument("--q", type=float, default=0.7)

    parser.add_argument(
        "--metadata_csv_name",
        type=str,
        default="metadata.csv",
        help="name of the csv data file (dataset csv has to be placed in dataset folder).",
    )
    parser.add_argument("--fold", default=None)
    # Our groups (upweighting/dro_ours)
    parser.add_argument(
        "--metadata_path",
        default=None,
        help="path to metadata csv",
    )
    parser.add_argument("--aug_col", default=None)

    # AUX2 input
    parser.add_argument("--use_weighted_spurious_score", default=False, action="store_true")
    parser.add_argument("--use_confidence", default=False, action="store_true")
    parser.add_argument("--confidence", default=0.5, type=float)
    parser.add_argument("--subsample_propotion", default=1.0, type=float)
    # parser.add_argument("--resume", default=False, action="store_true")

    # Process training put 
    parser.add_argument("--process_training", default=False, action="store_true")
    parser.add_argument("--best_epoch", type=int, default=0)
    parser.add_argument("--loadModel", type=str, default=None)
    parser.add_argument("--ProcessWhole", default=False, action="store_true")
    parser.add_argument("--load_new_model", action="store_true", default=False)
    # Loss type
    parser.add_argument("--loss", type=str, default="CrossEntropy", 
    help="GCE, CrossEntropy, SquaredLoss, LabelSmoothing")
    parser.add_argument("--minority_only", default=False, action="store_true")
    parser.add_argument("--val_subsample", type=float, default=1.0)

    # Control group
    parser.add_argument("--val_group0", type=int, default=467)
    parser.add_argument("--val_group1", type=int, default=466)
    parser.add_argument("--val_group2", type=int, default=133)
    parser.add_argument("--val_group3", type=int, default=133)
    parser.add_argument("--control_val_group", default=False, action="store_true")

    ###################### main args for thsi one 

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--short_text", type=int, default=21)
    parser.add_argument("--long_text", type=int, default=87)

    ######################
    args = parser.parse_args()
    divide_data(args)
    
    