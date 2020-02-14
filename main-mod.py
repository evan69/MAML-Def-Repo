import os
import json
import argparse

import torch
import numpy as np
from tensorboardX import SummaryWriter

from maml.datasets.omniglot import OmniglotMetaDataset
from maml.datasets.miniimagenet import MiniimagenetMetaDataset
from maml.datasets.cifar100 import Cifar100MetaDataset
from maml.datasets.bird import BirdMetaDataset
from maml.datasets.aircraft import AircraftMetaDataset
from maml.datasets.multimodal_few_shot import MultimodalFewShotDataset
from maml.models.fully_connected import FullyConnectedModel, MultiFullyConnectedModel
from maml.models.conv_net import ConvModel
from maml.models.gated_conv_net import GatedConvModel
from maml.models.gated_net import GatedNet
from maml.models.simple_embedding_model import SimpleEmbeddingModel
from maml.models.lstm_embedding_model import LSTMEmbeddingModel
from maml.models.gru_embedding_model import GRUEmbeddingModel
from maml.models.conv_embedding_model import ConvEmbeddingModel
from maml.metalearner import MetaLearner
from maml.trainer import Trainer
from maml.utils import optimizer_to_device, get_git_revision_hash

def main(args):
    is_training = not args.eval
    run_name = 'train' if is_training else 'eval'

    if is_training:
        writer = SummaryWriter('./train_dir/{0}/{1}'.format(
            args.output_folder, run_name))
        with open('./train_dir/{}/config.txt'.format(
            args.output_folder), 'w') as config_txt:
            for k, v in sorted(vars(args).items()):
                config_txt.write('{}: {}\n'.format(k, v))
    else:
        writer = None

    save_folder = './train_dir/{0}'.format(args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    config_name = '{0}_config.json'.format(run_name)
    with open(os.path.join(save_folder, config_name), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        try:
            config.update({'git_hash': get_git_revision_hash()})
        except:
            pass
        json.dump(config, f, indent=2)

    _num_tasks = 1

    # choose dataset
    if args.dataset == 'omniglot':
        dataset = OmniglotMetaDataset(
            root='data',
            img_side_len=28, # args.img_side_len,
            num_classes_per_batch=args.num_classes_per_batch,
            num_samples_per_class=args.num_samples_per_class,
            num_total_batches=args.num_batches,
            num_val_samples=args.num_val_samples,
            meta_batch_size=args.meta_batch_size,
            train=is_training,
            # num_train_classes=args.num_train_classes,
            num_workers=args.num_workers,
            device=args.device)
        loss_func = torch.nn.CrossEntropyLoss()
        collect_accuracies = True
    elif args.dataset == 'cifar':
        dataset = Cifar100MetaDataset(
            root='data',
            img_side_len=32,
            num_classes_per_batch=args.num_classes_per_batch,
            num_samples_per_class=args.num_samples_per_class,
            num_total_batches=args.num_batches,
            num_val_samples=args.num_val_samples,
            meta_batch_size=args.meta_batch_size,
            train=is_training,
            # num_train_classes=args.num_train_classes,
            num_workers=args.num_workers,
            device=args.device)
        loss_func = torch.nn.CrossEntropyLoss()
        collect_accuracies = True
    elif args.dataset == 'miniimagenet':
        dataset = MiniimagenetMetaDataset(
            root='data',
            img_side_len=84,
            num_classes_per_batch=args.num_classes_per_batch,
            num_samples_per_class=args.num_samples_per_class,
            num_total_batches=args.num_batches,
            num_val_samples=args.num_val_samples,
            meta_batch_size=args.meta_batch_size,
            train=is_training,
            # num_train_classes=args.num_train_classes,
            num_workers=args.num_workers,
            device=args.device)
        loss_func = torch.nn.CrossEntropyLoss()
        collect_accuracies = True
    else:
        raise ValueError('Unrecognized dataset {}'.format(args.dataset))

    # choose model
    if args.model_type == 'conv':
        model = ConvModel(
            input_channels=dataset.input_size[0],
            output_size=dataset.output_size,
            num_channels=args.num_channels,
            img_side_len=dataset.input_size[1],
            use_max_pool=args.use_max_pool,
            verbose=args.verbose)
    else:
        raise ValueError('Unrecognized model type {}'.format(args.model_type))

    # choose optimizers    
    model_parameters = list(model.parameters())
    optimizers = [torch.optim.Adam(model_parameters, lr=args.slow_lr)]

    # load checkpoint
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(args.device)

    # set meta learner
    meta_learner = MetaLearner(
        model, optimizers, fast_lr=args.fast_lr,
        loss_func=loss_func, first_order=args.first_order,
        num_updates=args.num_updates,
        inner_loop_grad_clip=args.inner_loop_grad_clip,
        collect_accuracies=collect_accuracies, device=args.device
        # alternating=args.alternating, embedding_schedule=args.embedding_schedule,
        # classifier_schedule=args.classifier_schedule, embedding_grad_clip=args.embedding_grad_clip
    )

    # set trainer
    trainer = Trainer(
        meta_learner=meta_learner, meta_dataset=dataset, writer=writer,
        log_interval=args.log_interval, save_interval=args.save_interval,
        model_type=args.model_type, save_folder=save_folder,
        total_iter=args.num_batches//args.meta_batch_size
    )

    if is_training:
        trainer.train()
    else:
        trainer.eval()

if __name__ == '__main__':

    def str2bool(arg):
        return arg.lower() == 'true'

    parser = argparse.ArgumentParser(
        description='Model-Agnostic Meta-Learning (MAML)')

    # Model
    parser.add_argument('--model-type', type=str, default='conv',
        help='type of the model')
    parser.add_argument('--use-max-pool', type=str2bool, default=False,
        help='choose whether to use max pooling with convolutional model')
    parser.add_argument('--num-channels', type=int, default=32,
        help='number of channels in convolutional layers')

    # Inner loop
    parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')
    parser.add_argument('--fast-lr', type=float, default=0.05,
        help='learning rate for the 1-step gradient update of MAML')
    parser.add_argument('--inner-loop-grad-clip', type=float, default=20.0,
        help='enable gradient clipping in the inner loop')
    parser.add_argument('--num-updates', type=int, default=5,
        help='how many update steps in the inner loop')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=1920000,
        help='number of batches')
    parser.add_argument('--meta-batch-size', type=int, default=10,
        help='number of tasks per batch')
    parser.add_argument('--slow-lr', type=float, default=0.001,
        help='learning rate for the global update of MAML')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--device', type=str, default='cuda',
        help='set the device (cpu or cuda)')
    parser.add_argument('--num-workers', type=int, default=4,
        help='how many DataLoader workers to use')
    parser.add_argument('--log-interval', type=int, default=100,
        help='number of batches between tensorboard writes')
    parser.add_argument('--save-interval', type=int, default=1000,
        help='number of batches between model saves')
    parser.add_argument('--eval', action='store_true', default=False,
        help='evaluate model')
    parser.add_argument('--checkpoint', type=str, default='',
        help='path to saved parameters.')

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar',
        help='which dataset to use')
    parser.add_argument('--data-root', type=str, default='data',
        help='path to store datasets')
    parser.add_argument('--num-classes-per-batch', type=int, default=5,
        help='how many classes per task')
    parser.add_argument('--num-samples-per-class', type=int, default=1,
        help='how many samples per class for training')
    parser.add_argument('--num-val-samples', type=int, default=15,
        help='how many samples per class for validation')
    parser.add_argument('--img-side-len', type=int, default=28,
        help='width and height of the input images')

    parser.add_argument('--verbose', type=str2bool, default=False,
        help='')

    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./train_dir'):
        os.makedirs('./train_dir')

    # args.model_type = 'conv'

    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')

    # print args
    if args.verbose:
        print('='*10 + ' ARGS ' + '='*10)
        for k, v in sorted(vars(args).items()):
            print('{}: {}'.format(k, v))
        print('='*26)

    main(args)
