import argparse
import os

import scipy.special

from dataManipulation import *
from utils import summary_raw, get_support_from_mcmc, namenum, tree_process, BitArray
from mixTreeBranchVBPI import mixTreeBranchVBPI
import numpy as np
import datetime
import json
import torch
import logging
import wandb
import ete3

parser = argparse.ArgumentParser()

parser.add_argument('--loggerName', type=str, default='main.log', help=' namn of the logging file')


######### Data arguments
parser.add_argument('--dataset', required=True, help=' DS1 | DS2 | DS3 | DS4 | DS5 | DS6 | DS7 | DS8 ')
parser.add_argument('--supportType', type=str, default='ufboot', help=' ufboot | _')

######### Model arguments
parser.add_argument('--psp', default=False, action='store_true', help=' turn on psp branch length feature')
parser.add_argument('--nf', type=int, default=2, help=' branch length feature embedding dimension ')
parser.add_argument('--test', default=False, action='store_true', help='turn on the test mode')
parser.add_argument('--train', default=False, action='store_true', help='turn on the training mode')
parser.add_argument('--datetime', type=str, default='2022-01-01', help=' 2020-04-01 | 2020-04-02 | ...... ')
parser.add_argument('--cf', type=int, default=-1, help=' checkpoint frequency ')
parser.add_argument('--ltf', type=int, default=5000, help=' test lower bound frequency ')


######### Learning parameters
parser.add_argument('--stepszTree', type=float, default=0.001, help=' step size for tree topology parameters ')
parser.add_argument('--stepszBranch', type=float, default=0.001, help=' stepsz for branch length parameters ')
parser.add_argument('--maxIter', type=int, default=400000, help=' number of iterations for training, default=400000')
parser.add_argument('--invT0', type=float, default=0.001, help=' initial inverse temperature for annealing schedule, default=0.001')
parser.add_argument('--nwarmStart', type=float, default=100000, help=' number of warm start iterations, default=100000')
parser.add_argument('--nParticle', type=int, default=10, help='number of particles for variational objectives, default=10')
parser.add_argument('--ar', type=float, default=0.75, help='step size anneal rate, default=0.75')
parser.add_argument('--af', type=int, default=20000, help='step size anneal frequency, default=20000')
parser.add_argument('--tf', type=int, default=1000, help='monitor frequency during training, default=1000')
parser.add_argument('--lbf', type=int, default=5000, help='lower bound test frequency, default=5000')
parser.add_argument('--gradMethod', type=str, default='vimco', help=' vimco | rws ')
parser.add_argument('--optimizer', type=str, default='adam', help=' adam | sgd ')

######### Load checkpoint and sample
parser.add_argument('--loadCheckpoint', type=str, default=None, help='./results/model.pt (point to main model)')
parser.add_argument('--resultPath', type=str, default="results/", help=' path to results folder')
parser.add_argument('--dataPath', type=str, default="data/", help=' path to data folder')
parser.add_argument('--goldenRunPath', type=str, default="data/goldenrun/", help=' path to mrbayes goldenrun')

####### NF
parser.add_argument('--flow_type', type=str, default='realnvp', help=' identity | planar | realnvp ')
parser.add_argument('--use_nf', default=False, action='store_true')
parser.add_argument('--sh', type=list, default=[100], help=' list of the hidden layer sizes for permutation invariant flow ')
parser.add_argument('--Lnf', type=int, default=10, help=' number of layers for permutation invariant flow ')

####### Support
parser.add_argument('--support_runs', type=list, default=[1,2,3,4,5,6,7,8,9,10], help="the numbers of runs used for support, can be used to speed up startup during development")
parser.add_argument('--ut_from_support', default=False, action='store_true', help="used to save topologies and variational likelihood of those in a json for all support trees")
parser.add_argument('--ut_from_goldenrun', default=False, action='store_true', help="used to save topologies and variational likelihood of those in a json for all trees in the goldenrun (require goldenRunPath)")
parser.add_argument('--goldenrun_hpd', type=float, default=1.0, help=' burnin')


####### Test
parser.add_argument('--test_n_particles',  type=int, default=1000)
parser.add_argument('--test_n_runs', type=int, default=100)
parser.add_argument('--test_opt_miselbo', default=False, action='store_true')

###### Settings for componenets optimization when finished
parser.add_argument('--stepszMISELBO', type=float, default=0.001)
parser.add_argument('--optMISELBO', type=str, default="sgd")
parser.add_argument('--itersMISELBO', type=int, default=10000)

parser.add_argument('--S', type=int, default=1, help="Amount of componenets in the mixture")
parser.add_argument('--mixture_type', type=str, default="multi", help="multi | multi_branch | multi_tree, used to only have mixtures in branches or trees")

####### Torch
parser.add_argument('--n_proc', type=int, default=2)

parser.add_argument('--wandb_mode', type=str, default="online", help='online | offline | disabled')
parser.add_argument('--wandb_group', type=str, default="dev", help='dev | run')
parser.add_argument('--wandb_prefix', type=str, default="", help='any string to start the runs name to help differentiate runs')

def main(args):
    torch.set_num_threads(args.n_proc)
    args.result_folder = os.path.join(args.resultPath, args.dataset)
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)

    args.save_to_path = args.result_folder + '/' + 'model.pt'

    LOG_DATEFMT = '%Y-%m-%d %H:%M:%S'
    LOG_FORMAT = ('[%(levelname)s/%(name)s:%(lineno)d] %(asctime)s ' +
                  '(%(processName)s/%(threadName)s)> %(message)s')
    logging.basicConfig(filename=os.path.join(args.result_folder, args.loggerName), filemode="w", level=logging.DEBUG, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
    logging.info("\n---------------------------------\n Starting new run \n---------------------------------")
    print(f"Logging to: {os.path.join(args.result_folder, args.loggerName)}")

    logging.info(f"Running with the following settings: {args}")

    name = f"{args.wandb_prefix}_vbpi_S{args.S}"
    if args.mixture_type: name+=f"_{args.mixture_type}"
    if args.use_nf: name+=f"_{args.flow_type}-{args.Lnf}"
    name+=f"_lrt{args.stepszTree}_lrb{args.stepszBranch}_{args.dataset}_{datetime.datetime.now()}"

    run = wandb.init(
        project="vbpi-mix",
        entity="name",
        name=name,
        mode=args.wandb_mode,
        group=args.wandb_group,
        magic=True,
        config=args
    )


    if args.dataset in ('DS' + str(i) for i in range(1, 11)):
        ufboot_support_path = os.path.join(args.dataPath, 'ufboot_data_DS1-11/')
        data_path = os.path.join(args.dataPath, 'hohna_datasets_fasta/')
        ground_truth_path, samp_size = os.path.join(args.dataPath, 'raw_data_DS1-11/'), 750001
    else:
        logging.warning("Dataset not supported...")

    logging.info(f"Loading Data set: {args.dataset}")

    data, taxa = loadData(data_path + args.dataset + '.fasta', 'fasta')


    if args.supportType == 'ufboot':

        tree_dict_support, tree_names_support = summary_raw(args.dataset, ufboot_support_path, runs=args.support_runs)
        rootsplit_supp_dict, subsplit_supp_dict = get_support_from_mcmc(taxa, tree_dict_support, tree_names_support)

        del tree_dict_support, tree_names_support

    model = mixTreeBranchVBPI(taxa, [rootsplit_supp_dict for _ in range(args.S)], [subsplit_supp_dict for _ in range(args.S)], data, pden=np.ones(4)/4., subModel=('JC', 1.0),
         feature_dim=args.nf,
         psp=args.psp,
         S=args.S,
         use_nf=args.use_nf,
         hidden_sizes=args.sh,
         num_of_layers_nf=args.Lnf,
         flow_type=args.flow_type,
         mixture_type=args.mixture_type)

    if args.loadCheckpoint:
        logging.info(f"Loading checkpoint at: {args.loadCheckpoint}")
        model.load_from(args.loadCheckpoint)

    # model.print_parameters()

    if args.train:
        logging.info('VBPI running, results will be saved to: {args.save_to_path}')

        test_lb = model.learn({'tree':args.stepszTree,'branch':args.stepszBranch}, args.maxIter,
                     test_freq=args.tf,
                     lb_test_freq=args.ltf,
                     n_particles=args.nParticle,
                     anneal_freq=args.af,
                     init_inverse_temp=args.invT0,
                     warm_start_interval=args.nwarmStart,
                     method=args.gradMethod,
                     optimizer=args.optimizer,
                     checkpoint_freq=args.cf,
                     save_to_path=args.save_to_path)

        np.save(args.save_to_path.replace('.pt', '_test_lb.npy'), test_lb)

    if args.test:

        if args.test_opt_miselbo:
            elbo_avg, elbo_std, miselbo_avg, miselbo_std, log_w_tilde = model.lower_bound_miselbo(stepz=args.stepszMISELBO, iters=args.itersMISELBO, opt=args.optMISELBO,n_particles=args.test_n_particles, n_runs=args.test_n_runs, opt_w=True)
            logging.info(f"Optim-MISELBO (mix): {miselbo_avg}±{miselbo_std} \t n_particles={args.test_n_particles}, n_runs={args.test_n_runs}")
            logging.info(f"Optim-ELBO (mix): {elbo_avg}±{elbo_std} \t n_particles={args.test_n_particles}, n_runs={args.test_n_runs}")

            wandb.log({"Optim Test-MISELBO-avg": miselbo_avg})
            wandb.log({"Optim Test-MISELBO-std": miselbo_std})
            wandb.log({"Optim Test-ELBO-avg": elbo_avg})
            wandb.log({"Optim Test-ELBO-std": elbo_std})
            for s in range(args.S):
                wandb.log({f"W_s={s}": torch.exp(log_w_tilde)[s].item()})

            elbo_avg, elbo_std, miselbo_avg, miselbo_std, log_w_tilde = model.lower_bound_miselbo( n_particles=args.test_n_particles, n_runs=args.test_n_runs, opt_w=False)
            logging.info(f"Test-MISELBO (mix): {miselbo_avg}±{miselbo_std} \t n_particles={args.test_n_particles}, n_runs={args.test_n_runs}")
            wandb.log({"Final Test-MISELBO-avg": miselbo_avg})
            wandb.log({"Final Test-MISELBO-std": miselbo_std})

        else:
            elbo_avg, elbo_std = model.lower_bound( n_particles=args.test_n_particles, n_runs=args.test_n_runs)

        logging.info(f"Test-ELBO (regular/mix): {elbo_avg}±{elbo_std} \t n_particles={args.test_n_particles}, n_runs={args.test_n_runs}")
        wandb.log({"Final Test-ELBO-avg": elbo_avg})
        wandb.log({"Final Test-ELBO-std": elbo_std})

    if args.ut_from_support:
        logging.info("Creating dict with q(τ) for all support trees")
        translation_d2n = {i + 1: model.taxa[i] for i in range(len(model.taxa))}
        with open(os.path.join(args.result_folder, "translation_d2n.json"), "w") as f:
            f.write(json.dumps(translation_d2n))

        with torch.no_grad():
                logging.info("Start creating q(τ) from support")
                tree_dict_support, tree_names_support = summary_raw(args.dataset, ufboot_support_path,
                                                                    runs=args.support_runs)
                logging.info(f"Amount of trees in support: {len(tree_dict_support)}")

                toBitArr = BitArray(taxa)
                for n,t in tree_dict_support.items():
                    tree_process(t, toBitArr)
                    namenum(t, model.taxa)

                unique_tree_dict = {}
                for _,t in tree_dict_support.items():
                    logqs = [model.log_w_tilde[s].item() + model.logq_tree_s(s, t).item() for s in range(len(model.tree_model))]
                    logqs_mix = scipy.special.logsumexp(logqs)

                    for leaf in t.get_leaves():
                        leaf.name = str(int(leaf.name) + 1)

                    nw_tree = t.write(format=9)
                    t = ete3.Tree(nw_tree)  # making sure the format is the same
                    id = t.get_topology_id()
                    unique_tree_dict[id] = [nw_tree, logqs_mix, *logqs] #tree, q(τ), q_1(τ),...,q_S(τ)

                with open(args.result_folder + f'/unique_trees_mix.json', 'w') as outfile:
                    json.dump(unique_tree_dict, outfile)

    if args.ut_from_goldenrun:
        logging.info("Creating dict with q(τ) for all goldenrun trees")
        translation_d2n = {i + 1: model.taxa[i] for i in range(len(model.taxa))}
        with open(os.path.join(args.result_folder, "translation_d2n.json"), "w") as f:
            f.write(json.dumps(translation_d2n))

        with torch.no_grad():
                logging.info("Start creating q(τ) from support")
                tree_dict_support, tree_names_support = summary_raw(args.dataset, args.goldenRunPath,
                                                                    runs=args.support_runs, goldenrun=True, translation_d2n=translation_d2n, hpd=args.goldenrun_hpd)

                logging.info(f"Amount of trees in support: {len(tree_dict_support)}")

                toBitArr = BitArray(taxa)
                for n,t in tree_dict_support.items():
                    tree_process(t, toBitArr)
                    namenum(t, model.taxa)


                unique_tree_dict = {}
                for _,t in tree_dict_support.items():
                    logqs = [model.log_w_tilde[s].item() + model.logq_tree_s(s, t).item() for s in range(len(model.tree_model))]
                    logqs_mix = scipy.special.logsumexp(logqs)

                    for leaf in t.get_leaves():
                        leaf.name = str(int(leaf.name) + 1)

                    nw_tree = t.write(format=9)
                    t = ete3.Tree(nw_tree)  # making sure the format is the same
                    id = t.get_topology_id()
                    unique_tree_dict[id] = [nw_tree, logqs_mix, *logqs] #tree, q(τ), q_1(τ),...,q_S(τ)

                with open(args.result_folder + f'/unique_trees_mix.json', 'w') as outfile:
                    json.dump(unique_tree_dict, outfile)

    return model

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
