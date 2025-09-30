import argparse
import logging
import time
import os


def parse_train_args(parser):

    # Global
    parser.add_argument('--save_log_dir', type=str, default='./logs_tmp')
    parser.add_argument('--best_source_dir', type=str, default='./best_source/freezes')
    parser.add_argument('--best_target_dir', type=str, default='./best_target/')
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument("--model_name", type=str, default="temp")
    parser.add_argument('--randn', type=int, default=1110)
    # UDA
    parser.add_argument('--source_dataset', type=str, default='TW-20', help='name of dataset')
    parser.add_argument('--target_dataset', type=str, default='C-15', help='name of dataset')
    parser.add_argument('--category_feature_size', type=int, default=3, help='')
    parser.add_argument('--numerical_feature_size', type=int, default=5, help='')
    parser.add_argument('--textual_feature_size', type=int, default=768, help='')

    # Model
    parser.add_argument('--agg_strategy', type=str, default="mlp", choices=['mlp', 'att'], help='feature fuse strategy')
    parser.add_argument('--feature_dim', type=int, default=528, help='Hidden dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--gnn_layers', type=int, default=2, help='number of layers of gnn network')
    parser.add_argument('--gnn_model', type=str, default="HGT", help='Select different model')
    parser.add_argument('--head_nums', type=int, default=4, help="")
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--classifier_out', type=int, default=2)
    parser.add_argument('--batch_norm', type=int, default=0, choices=[0, 1], help='use batch norm layer in GNN')

    parser.add_argument('--with_adapt', type=int, default=1, choices=[0, 1], help='')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--sparse', type=int, default=0)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--proj_dim', type=int, default=128)      
    parser.add_argument('--extra_feature_size', type=int, default=256)
    parser.add_argument('--cl_batch_size', type=int, default=0)    
    parser.add_argument('--k', type=int, default=20)                 
    parser.add_argument('--maskfeat_rate_1', type=float, default=0.3)
    parser.add_argument('--maskfeat_rate_2', type=float, default=0.3)
    parser.add_argument('--dropedge_rate_1', type=float, default=0.3)
    parser.add_argument('--dropedge_rate_2', type=float, default=0.3)
    parser.add_argument('--lamb', type=float, default=0.5, help='trade-off parameter lambda')
    parser.add_argument('--gama', type=float, default=0.5, help='uda trade-off parameter ')

    # Train
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (# nodes)')
    parser.add_argument('--source_weight_decay0', type=float, default=5e-4, help='')
    parser.add_argument('--source_lr0', type=float, default=0.001, help='input learning rate for training')
    parser.add_argument('--source_weight_decay1', type=float, default=5e-4, help='')
    parser.add_argument('--source_lr1', type=float, default=0.001, help='input learning rate for training')
    parser.add_argument('--target_weight_decay0', type=float, default=5e-4, help='')
    parser.add_argument('--target_lr0', type=float, default=0.00001, help='input learning rate for training')
    parser.add_argument('--target_weight_decay1', type=float, default=5e-4, help='')
    parser.add_argument('--target_lr1', type=float, default=0.00001, help='')
    parser.add_argument('--source_epochs', type=int, default=80, help='')
    parser.add_argument('--loop_0', type=int, default=1, help='extra model loop')
    parser.add_argument('--loop_1', type=int, default=1, help='baseline model loop')
    parser.add_argument('--target_epochs', type=int, default=20, help='')
    parser.add_argument('--reset_model', type=int, default=0, help='Reset unsupervised model', choices=[0, 1])
    parser.add_argument('--with_extra', type=int, default=0, help='Whether to use D-VGA', choices=[0, 1])
    parser.add_argument('--with_activate', type=int, default=0, help='', choices=[0, 1])

    parser.add_argument('--struct_lambda', type=float, default=1, help='Structure NCE loss')
    parser.add_argument('--neigh_lambda', type=float, default=1, help='Neighborhood NCE loss')
    parser.add_argument('--im_lambda', type=float, default=1, help='IM loss')

    parser.add_argument('--momentum', type=float, default=0.1, help='momentum')

    return parser


def init_logger(args):
    log_file = "{}/Bot_src-{}_trg-{}_seed-{}_gnn-{}_layer-{}_feat-{}_dim-{}_drp-{}_{}.log".format(
        args.save_log_dir,
        args.source_dataset,
        args.target_dataset,
        args.random_seed,
        args.gnn_model,
        args.gnn_layers,
        args.agg_strategy,
        args.hidden_dim,
        args.dropout,
        "tmp"
    )
    if not os.path.exists(args.save_log_dir):
        os.mkdir(args.save_log_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("logger name:%s", args.model_name + ".log")
    return logger