import os
from datetime import datetime

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import ConcatDataset
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
import numpy as np

from models.vcl_nn import DiscriminativeVCL
from models.coreset import RandomCoreset
from util.experiment_utils import run_point_estimate_initialisation, run_task
from util.transforms import Flatten, Scale, Permute, Permute2D
from util.datasets import NOTMNIST

USER = os.environ['USER']

MNIST_FLATTENED_DIM = 28 * 28
LR = 0.001
INITIAL_POSTERIOR_VAR = 1e-3

CIFAR_DIM = 3 * 32 * 32  # CIFAR10 images are 32x32 pixels with 3 color channels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device", device)


def permuted_cifar10():
    N_CLASSES = 10  # For CIFAR10's 10 classes
    LAYER_WIDTH = 256
    N_HIDDEN_LAYERS = 2
    N_TASKS = 10
    MULTIHEADED = False
    CORESET_SIZE = 200
    EPOCHS = 10
    BATCH_SIZE = 1024
    TRAIN_FULL_CORESET = True

    # Normalization for CIFAR10
    normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # Define the Flatten, Scale (Normalize here), and Permute transformations
    transforms = [Compose([
        Permute2D(torch.randperm(CIFAR_DIM // 3)),
        normalize,
        Flatten()
    ]) for _ in range(N_TASKS)]

    model = DiscriminativeVCL(
        in_size=CIFAR_DIM, out_size=N_CLASSES,
        layer_width=LAYER_WIDTH, n_hidden_layers=N_HIDDEN_LAYERS,
        n_heads=(N_TASKS if MULTIHEADED else 1),
        initial_posterior_var=INITIAL_POSTERIOR_VAR
    ).to(device)

    coreset = RandomCoreset(size=CORESET_SIZE)

    # Load CIFAR10 datasets with the defined transforms for each task
    cifar_train = ConcatDataset(
        [CIFAR10(root="data", train=True, download=False, transform=t) for t in transforms]
    )
    task_size = len(cifar_train) // N_TASKS
    train_task_ids = torch.cat(
        [torch.full((task_size,), id) for id in range(N_TASKS)]
    )

    cifar_test = ConcatDataset(
        [CIFAR10(root="data", train=False, download=False, transform=t) for t in transforms]
    )
    task_size = len(cifar_test) // N_TASKS
    test_task_ids = torch.cat(
        [torch.full((task_size,), id) for id in range(N_TASKS)]
    )

    summary_logdir = os.path.join("logs", "disc_p_cifar10", datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(summary_logdir)

    run_point_estimate_initialisation(model=model, data=cifar_train,
                                      epochs=EPOCHS, batch_size=BATCH_SIZE,
                                      device=device, lr=LR,
                                      multiheaded=MULTIHEADED,
                                      task_ids=train_task_ids)

    for task in range(N_TASKS):
        run_task(
            model=model, train_data=cifar_train, train_task_ids=train_task_ids,
            test_data=cifar_test, test_task_ids=test_task_ids, task_idx=task,
            coreset=coreset, epochs=EPOCHS, batch_size=BATCH_SIZE,
            device=device, lr=LR, save_as=f"disc_p_cifar10_{{datetime.now().strftime('%b%d_%H-%M-%S')}}",
            multiheaded=MULTIHEADED, train_full_coreset=TRAIN_FULL_CORESET,
            summary_writer=writer
        )

    writer.close()



def split_cifar10():
    """
    Runs the 'Split CIFAR10' experiment, in which each task is
    a binary classification task carried out on a subset of the CIFAR10 dataset.
    """
    N_CLASSES = 2 
    LAYER_WIDTH = 128
    N_HIDDEN_LAYERS = 2
    N_TASKS = 5
    MULTIHEADED = True
    CORESET_SIZE = 200
    EPOCHS = 80
    BATCH_SIZE = 256
    TRAIN_FULL_CORESET = False

    # Normalization for CIFAR10
    normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform = Compose([
        ToTensor(),
        normalize,
        Flatten()
    ])

    # download dataset
    cifar_train = CIFAR10(root="data", train=True, download=False, transform=transform)
    cifar_test = CIFAR10(root="data", train=False, download=False, transform=transform)

    model = DiscriminativeVCL(
        in_size=CIFAR_DIM, out_size=N_CLASSES,
        layer_width=LAYER_WIDTH, n_hidden_layers=N_HIDDEN_LAYERS,
        n_heads=(N_TASKS if MULTIHEADED else 1),
        initial_posterior_var=INITIAL_POSTERIOR_VAR
    ).to(device)

    coreset = RandomCoreset(size=CORESET_SIZE)

    label_to_task_mapping = {
        0: 0, 1: 0,
        2: 1, 3: 1,
        4: 2, 5: 2,
        6: 3, 7: 3,
        8: 4, 9: 4,
    }

    if isinstance(cifar_train[0][1], int):
        train_task_ids = torch.Tensor([label_to_task_mapping[y] for _, y in cifar_train])
        test_task_ids = torch.Tensor([label_to_task_mapping[y] for _, y in cifar_test])
    elif isinstance(cifar_train[0][1], torch.Tensor):
        train_task_ids = torch.Tensor([label_to_task_mapping[y.item()] for _, y in cifar_train])
        test_task_ids = torch.Tensor([label_to_task_mapping[y.item()] for _, y in cifar_test])

    summary_logdir = os.path.join("logs", "disc_s_cifar10", datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(summary_logdir)

    # each task is a binary classification task for a different pair of digits
    binarize_y = lambda y, task: (y == (2 * task + 1)).long()

    run_point_estimate_initialisation(model=model, data=cifar_train,
                                      epochs=EPOCHS, batch_size=BATCH_SIZE,
                                      device=device, multiheaded=MULTIHEADED,
                                      lr=LR, task_ids=train_task_ids,
                                      y_transform=binarize_y)

    for task_idx in range(N_TASKS):
        run_task(
            model=model, train_data=cifar_train, train_task_ids=train_task_ids,
            test_data=cifar_test, test_task_ids=test_task_ids, coreset=coreset,
            task_idx=task_idx, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
            save_as=f"disc_s_cifar10_{datetime.now().strftime('%b%d_%H-%M-%S')}_coreset{CORESET_SIZE}", device=device, multiheaded=MULTIHEADED,
            y_transform=binarize_y, train_full_coreset=TRAIN_FULL_CORESET,
            summary_writer=writer
        )

    writer.close()

def split_cifar100():
    """
    Runs the 'Split CIFAR100' experiment, in which each task is
    a binary classification task carried out on a subset of the CIFAR10 dataset.
    """
    N_CLASSES = 10
    LAYER_WIDTH = 256
    N_HIDDEN_LAYERS = 2
    N_TASKS = 10
    MULTIHEADED = True
    CORESET_SIZE = 100
    EPOCHS = 200
    BATCH_SIZE = 256
    TRAIN_FULL_CORESET = True

    # Normalization for CIFAR10
    normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # permutation used for each task
    transform = Compose([
        ToTensor(),
        normalize,
        Flatten()
    ])

    # download dataset
    cifar_train = CIFAR100(root="data", train=True, download=False, transform=transform)
    cifar_test = CIFAR100(root="data", train=False, download=False, transform=transform)

    model = DiscriminativeVCL(
        in_size=CIFAR_DIM, out_size=N_CLASSES,
        layer_width=LAYER_WIDTH, n_hidden_layers=N_HIDDEN_LAYERS,
        n_heads=(N_TASKS if MULTIHEADED else 1),
        initial_posterior_var=INITIAL_POSTERIOR_VAR
    ).to(device)

    coreset = RandomCoreset(size=CORESET_SIZE)

    label_to_task_mapping = lambda label: label // (100 // N_TASKS)

    train_task_ids = torch.Tensor([label_to_task_mapping(y) for _, y in cifar_train])
    test_task_ids = torch.Tensor([label_to_task_mapping(y) for _, y in cifar_test])

    summary_logdir = os.path.join("logs", "disc_s_cifar100", datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(summary_logdir)

    # each task is a binary classification task for a different pair of digits
    map_to_under_ten = lambda y, task: y % N_CLASSES

    run_point_estimate_initialisation(model=model, data=cifar_train,
                                      epochs=EPOCHS, batch_size=BATCH_SIZE,
                                      device=device, multiheaded=MULTIHEADED,
                                      lr=LR, task_ids=train_task_ids,
                                      y_transform=map_to_under_ten)

    for task_idx in range(N_TASKS):
        run_task(
            model=model, train_data=cifar_train, train_task_ids=train_task_ids,
            test_data=cifar_test, test_task_ids=test_task_ids, coreset=coreset,
            task_idx=task_idx, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
            save_as=f"disc_s_cifar100_{datetime.now().strftime('%b%d_%H-%M-%S')}_coreset{CORESET_SIZE}", device=device, multiheaded=MULTIHEADED,
            y_transform=map_to_under_ten, train_full_coreset=TRAIN_FULL_CORESET,
            summary_writer=writer
        )

    writer.close()


def permuted_mnist():
    """
    Runs the 'Permuted MNIST' experiment from the VCL paper, in which each task
    is obtained by applying a fixed random permutation to the pixels of each image.
    """
    N_CLASSES = 10
    LAYER_WIDTH = 100
    N_HIDDEN_LAYERS = 2
    N_TASKS = 10
    MULTIHEADED = False
    CORESET_SIZE = 200
    EPOCHS = 100
    BATCH_SIZE = 256
    TRAIN_FULL_CORESET = True

    # flattening and permutation used for each task
    transforms = [Compose([Flatten(), Scale(), Permute(torch.randperm(MNIST_FLATTENED_DIM))]) for _ in range(N_TASKS)]

    # create model, single-headed in permuted MNIST experiment
    model = DiscriminativeVCL(
        in_size=MNIST_FLATTENED_DIM, out_size=N_CLASSES,
        layer_width=LAYER_WIDTH, n_hidden_layers=N_HIDDEN_LAYERS,
        n_heads=(N_TASKS if MULTIHEADED else 1),
        initial_posterior_var=INITIAL_POSTERIOR_VAR
    ).to(device)
    coreset = RandomCoreset(size=CORESET_SIZE)

    mnist_train = ConcatDataset(
        [MNIST(root="data", train=True, download=True, transform=t) for t in transforms]
    )
    task_size = len(mnist_train) // N_TASKS
    train_task_ids = torch.cat(
        [torch.full((task_size,), id) for id in range(N_TASKS)]
    )

    mnist_test = ConcatDataset(
        [MNIST(root="data", train=False, download=True, transform=t) for t in transforms]
    )
    task_size = len(mnist_test) // N_TASKS
    test_task_ids = torch.cat(
        [torch.full((task_size,), id) for id in range(N_TASKS)]
    )

    summary_logdir = os.path.join("logs", "disc_p_mnist", datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(summary_logdir)
    run_point_estimate_initialisation(model=model, data=mnist_train,
                                      epochs=EPOCHS, batch_size=BATCH_SIZE,
                                      device=device, lr=LR,
                                      multiheaded=MULTIHEADED,
                                      task_ids=train_task_ids)

    # each task is classification of MNIST images with permuted pixels
    for task in range(N_TASKS):
        run_task(
            model=model, train_data=mnist_train, train_task_ids=train_task_ids,
            test_data=mnist_test, test_task_ids=test_task_ids, task_idx=task,
            coreset=coreset, epochs=EPOCHS, batch_size=BATCH_SIZE,
            device=device, lr=LR, save_as="disc_p_mnist",
            multiheaded=MULTIHEADED, train_full_coreset=TRAIN_FULL_CORESET,
            summary_writer=writer
        )

    writer.close()


def split_mnist():
    """
    Runs the 'Split MNIST' experiment from the VCL paper, in which each task is
    a binary classification task carried out on a subset of the MNIST dataset.
    """
    N_CLASSES = 2 # TODO does it make sense to do binary classification with out_size=2 ?
    LAYER_WIDTH = 256
    N_HIDDEN_LAYERS = 2
    N_TASKS = 5
    MULTIHEADED = True
    CORESET_SIZE = 40
    EPOCHS = 40
    BATCH_SIZE = 256
    TRAIN_FULL_CORESET = True

    transform = Compose([Flatten(), Scale()])

    # download dataset
    mnist_train = MNIST(root="data", train=True, download=True, transform=transform)
    mnist_test = MNIST(root="data", train=False, download=True, transform=transform)

    model = DiscriminativeVCL(
        in_size=MNIST_FLATTENED_DIM, out_size=N_CLASSES,
        layer_width=LAYER_WIDTH, n_hidden_layers=N_HIDDEN_LAYERS,
        n_heads=(N_TASKS if MULTIHEADED else 1),
        initial_posterior_var=INITIAL_POSTERIOR_VAR
    ).to(device)

    coreset = RandomCoreset(size=CORESET_SIZE)

    label_to_task_mapping = {
        0: 0, 1: 0,
        2: 1, 3: 1,
        4: 2, 5: 2,
        6: 3, 7: 3,
        8: 4, 9: 4,
    }

    if isinstance(mnist_train[0][1], int):
        train_task_ids = torch.Tensor([label_to_task_mapping[y] for _, y in mnist_train])
        test_task_ids = torch.Tensor([label_to_task_mapping[y] for _, y in mnist_test])
    elif isinstance(mnist_train[0][1], torch.Tensor):
        train_task_ids = torch.Tensor([label_to_task_mapping[y.item()] for _, y in mnist_train])
        test_task_ids = torch.Tensor([label_to_task_mapping[y.item()] for _, y in mnist_test])

    summary_logdir = os.path.join("logs", "disc_s_mnist", datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(summary_logdir)

    # each task is a binary classification task for a different pair of digits
    binarize_y = lambda y, task: (y == (2 * task + 1)).long()

    run_point_estimate_initialisation(model=model, data=mnist_train,
                                      epochs=EPOCHS, batch_size=BATCH_SIZE,
                                      device=device, multiheaded=MULTIHEADED,
                                      lr=LR, task_ids=train_task_ids,
                                      y_transform=binarize_y)

    for task_idx in range(N_TASKS):
        run_task(
            model=model, train_data=mnist_train, train_task_ids=train_task_ids,
            test_data=mnist_test, test_task_ids=test_task_ids, coreset=coreset,
            task_idx=task_idx, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
            save_as="disc_s_mnist", device=device, multiheaded=MULTIHEADED,
            y_transform=binarize_y, train_full_coreset=TRAIN_FULL_CORESET,
            summary_writer=writer
        )

    writer.close()


def split_not_mnist():
    """
    Runs the 'Split not MNIST' experiment from the VCL paper, in which each task
    is a binary classification task carried out on a subset of the not MNIST
    character recognition dataset.
    """
    N_CLASSES = 2 # TODO does it make sense to do binary classification with out_size=2 ?
    LAYER_WIDTH = 150
    N_HIDDEN_LAYERS = 4
    N_TASKS = 5
    MULTIHEADED = True
    CORESET_SIZE = 40
    EPOCHS = 120
    BATCH_SIZE = 400000
    TRAIN_FULL_CORESET = True

    transform = Compose([Flatten(), Scale()])

    not_mnist_train = NOTMNIST(train=True, overwrite=False, transform=transform)
    not_mnist_test = NOTMNIST(train=False, overwrite=False, transform=transform)

    model = DiscriminativeVCL(
        in_size=MNIST_FLATTENED_DIM, out_size=N_CLASSES,
        layer_width=LAYER_WIDTH, n_hidden_layers=N_HIDDEN_LAYERS,
        n_heads=(N_TASKS if MULTIHEADED else 1),
        initial_posterior_var=INITIAL_POSTERIOR_VAR
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    coreset = RandomCoreset(size=CORESET_SIZE)

    # The y classes are integers 0-9.
    label_to_task_mapping = {
        0: 0, 1: 1,
        2: 2, 3: 3,
        4: 4, 5: 0,
        6: 1, 7: 2,
        8: 3, 9: 4,
    }

    train_task_ids = torch.from_numpy(np.array([label_to_task_mapping[y] for _, y in not_mnist_train]))
    test_task_ids = torch.from_numpy(np.array([label_to_task_mapping[y] for _, y in not_mnist_test]))

    summary_logdir = os.path.join("logs", "disc_s_n_mnist", datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(summary_logdir)

    # each task is a binary classification task for a different pair of digits
    # binarize_y(c, n) is 1 when c is is the nth digit - A for task 0, B for task 1
    binarize_y = lambda y, task: (y == task).long()

    run_point_estimate_initialisation(model=model, data=not_mnist_train,
                                      epochs=EPOCHS, batch_size=BATCH_SIZE,
                                      device=device, multiheaded=MULTIHEADED,
                                      task_ids=train_task_ids, lr=LR,
                                      y_transform=binarize_y)

    for task_idx in range(N_TASKS):
        run_task(
            model=model, train_data=not_mnist_train, train_task_ids=train_task_ids,
            test_data=not_mnist_test, test_task_ids=test_task_ids,
            coreset=coreset, task_idx=task_idx, epochs=EPOCHS, lr=LR,
            batch_size=BATCH_SIZE, save_as="disc_s_n_mnist", device=device,
            multiheaded=MULTIHEADED, y_transform=binarize_y,
            train_full_coreset=TRAIN_FULL_CORESET, summary_writer=writer
        )

    writer.close()
