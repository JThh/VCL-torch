import argparse
from collections import OrderedDict
import experiments.discriminative
import experiments.discriminative_conv
import experiments.discriminative_resnet
import experiments.generative


# experiments from the VCL paper that can be carried out
EXP_OPTIONS = {
    'disc_p_mnist': experiments.discriminative.permuted_mnist,
    'disc_s_mnist': experiments.discriminative.split_mnist,
    'disc_s_cifar10': experiments.discriminative.split_cifar10,
    'disc_s_cifar100': experiments.discriminative.split_cifar100,
    'disc_conv_p_mnist': experiments.discriminative_conv.permuted_mnist,
    'disc_resnet_p_cifar10': experiments.discriminative_resnet.permuted_cifar10,
    'disc_resnet_s_cifar10': experiments.discriminative_resnet.split_cifar10,
    'disc_resnet_s_cifar100': experiments.discriminative_resnet.split_cifar100,
    'disc_resnet_s_n_mnist': experiments.discriminative_resnet.split_not_mnist,
    'disc_resnet_s_mnist': experiments.discriminative_resnet.split_mnist,
    'disc_resnet_p_mnist': experiments.discriminative_resnet.permuted_mnist,
    'disc_conv_p_cifar10': experiments.discriminative_conv.permuted_cifar10,
    # 'disc_conv_large_p_cifar10': experiments.discriminative_conv.permuted_cifar10,  # more layers; now propagated to other cases as well.
    'disc_conv_s_mnist': experiments.discriminative_conv.split_mnist,
    'disc_conv_s_n_mnist': experiments.discriminative_conv.split_not_mnist,
    'disc_conv_s_cifar10': experiments.discriminative_conv.split_cifar10,
    'disc_conv_s_cifar100': experiments.discriminative_conv.split_cifar100,
    # 'disc_pconv_s_cifar10': experiments.discriminative_conv.split_cifar10,  # partial conv; see PartialConvVcl implementation at ~/src/models/contrib.py
    'disc_s_n_mnist': experiments.discriminative.split_not_mnist,
    'disc_p_cifar10': experiments.discriminative.permuted_cifar10,
    'gen_mnist_classifier': experiments.generative.train_mnist_classifier,
    'gen_n_mnist_classifier': experiments.generative.train_not_mnist_classifier,
    'gen_mnist': experiments.generative.generate_mnist,
    'gen_not_mnist': experiments.generative.generate_not_mnist,
}

EXP_OPTIONS = OrderedDict(EXP_OPTIONS)

def main(experiment='all', epochs=200):
    # run all experiments
    if experiment == 'all':
        for exp in list(EXP_OPTIONS.keys()):
            print("Running", exp)
            EXP_OPTIONS[exp]()
    else:
        EXP_OPTIONS[experiment]()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('experiment', help='Experiment to be run, can be one of: ' + str(list(EXP_OPTIONS.keys())))
    args = parser.parse_args()
    main(args.experiment)
