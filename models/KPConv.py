import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import numpy as np
from .architectures import *
from os import makedirs, remove
from os.path import exists, join
from data.SemanticKitti import SemanticKittiDataset, SemanticKittiSampler, SemanticKittiCollate
from utils.metrics import IoU_from_confusions, fast_confusion
from torch.optim.lr_scheduler import LambdaLR
from utils.config import Config
from torch.utils.data import DataLoader

class SemanticKittiConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'SemanticKitti'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 10

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']

    ###################
    # KPConv parameters
    ###################

    # Radius of the input sphere
    in_radius = 4.0
    val_radius = 4.0
    n_frames = 1
    max_in_points = 100000
    max_val_points = 100000

    # Number of batch
    batch_num = 8
    val_batch_num = 8

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.06

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 6.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 128
    in_features_dim = 2

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.02

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 800

    # Learning rate management
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1 / 150) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of steps per epochs
    epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = 200

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_color = 0.8

    # Choose weights for class (used in segmentation loss). Empty list for no weights
    # class proportion for R=10.0 and dl=0.08 (first is unlabeled)
    # 19.1 48.9 0.5  1.1  5.6  3.6  0.7  0.6  0.9 193.2 17.7 127.4 6.7 132.3 68.4 283.8 7.0 78.5 3.3 0.8
    #
    #

    # sqrt(Inverse of proportion * 100)
    # class_w = [1.430, 14.142, 9.535, 4.226, 5.270, 11.952, 12.910, 10.541, 0.719,
    #            2.377, 0.886, 3.863, 0.869, 1.209, 0.594, 3.780, 1.129, 5.505, 11.180]

    # sqrt(Inverse of proportion * 100)  capped (0.5 < X < 5)
    # class_w = [1.430, 5.000, 5.000, 4.226, 5.000, 5.000, 5.000, 5.000, 0.719, 2.377,
    #            0.886, 3.863, 0.869, 1.209, 0.594, 3.780, 1.129, 5.000, 5.000]

    # Do we nee to save convergence
    saving = True
    saving_path = None

    gpus = [0]
            
    output_dir = "./output"
    distributed_backend = "ddp"
    num_sanity_val_steps = 0

    checksequence = False

class LightningNetwork(pl.LightningModule):
    def __init__(self, configs: dict):
        super().__init__()
        self.initilization()
        self.config = SemanticKittiConfig()
        self.checksequence = self.config.checksequence
        print('\nModel Preparation')
        print('*****************')
        self.load_dataset_sampler()
        # Define network model
        t1 = time.time()
        training_dataset = self.train_dataloader().dataset
        self.net = KPFCNN(self.config, training_dataset.label_values, training_dataset.ignored_labels)
        self.debug = False
        if self.debug:
            print('\n*************************************\n')
            print(self.net)
            print('\n*************************************\n')
            for param in self.net.parameters():
                if param.requires_grad:
                    print(param.shape)
            print('\n*************************************\n')
            print("Model size %i" % sum(param.numel() for param in self.net.parameters() if param.requires_grad))
            print('\n*************************************\n')
        self.load_checkpoints(finetune = None)
        self.load_dataset_sampler()

    def initilization(self):
        ############################
        # Initialize the environment
        ############################

        # Set which gpu is going to be used
        GPU_ID = '0'

        # Set GPU visible device
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

        ###############
        # Previous chkp
        ###############

        # Choose here if you want to start training from a previous snapshot (None for new training)
        # previous_training_path = 'Log_2020-03-19_19-53-27'
        previous_training_path = ''

        # Choose index of checkpoint to start from. If None, uses the latest chkp
        chkp_idx = None

        if previous_training_path:

            # Find all snapshot in the chosen training folder
            chkp_path = os.path.join('results', previous_training_path, 'checkpoints')
            chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

            # Find which snapshot to restore
            if chkp_idx is None:
                chosen_chkp = 'current_chkp.tar'
            else:
                chosen_chkp = np.sort(chkps)[chkp_idx]
            self.chosen_chkp = os.path.join('results', previous_training_path, 'checkpoints', chosen_chkp)

        else:
            self.chosen_chkp = None
    
    def load_checkpoints(self, finetune = None):
        if (self.chosen_chkp is not None):
            if finetune:
                checkpoint = torch.load(self.chosen_chkp)
                self.net.load_state_dict(checkpoint['model_state_dict'])
                print("Model restored and ready for finetuning.")
            else:
                checkpoint = torch.load(self.chosen_chkp)
                self.net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.current_epoch = checkpoint['epoch']
                print("Model and training state restored.")

        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                self.config.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            if not exists(self.config.saving_path):
                makedirs(self.config.saving_path)
            self.config.save()

    def load_dataset_sampler(self):
        self.training_dataset = SemanticKittiDataset(self.config, set='training',
                                                balance_classes=True)
        self.training_sampler = SemanticKittiSampler(self.training_dataset)
    
        self.training_loader = DataLoader(self.training_dataset,
                                 batch_size=1,
                                 sampler=self.training_sampler,
                                 collate_fn=SemanticKittiCollate,
                                 num_workers=self.config.input_threads,
                                 pin_memory=True)
        self.training_sampler.calib_max_in(self.config, self.training_loader, verbose=True)
        self.training_sampler.calibration(self.training_loader, verbose=True)
        self.test_dataset = SemanticKittiDataset(self.config, set='validation',
                                            balance_classes=False)
        self.test_sampler = SemanticKittiSampler(self.test_dataset)
        self.test_loader = DataLoader(self.test_dataset,
                             batch_size=1,
                             sampler=self.test_sampler,
                             collate_fn=SemanticKittiCollate,
                             num_workers=self.config.input_threads,
                             pin_memory=True)
        self.test_sampler.calib_max_in(self.config, self.test_loader, verbose=True)
        self.test_sampler.calibration(self.test_loader, verbose=True)

    def train_dataloader(self):
        # self.training_dataset = SemanticKittiDataset(self.config, set='training',
        #                                         balance_classes=True)
        # self.training_sampler = SemanticKittiSampler(self.training_dataset)

        # training_loader = DataLoader(self.training_dataset,
        #                          batch_size=1,
        #                          sampler=self.training_sampler,
        #                          collate_fn=SemanticKittiCollate,
        #                          num_workers=self.config.input_threads,
        #                          pin_memory=True)
        # self.training_sampler.calib_max_in(self.config, training_loader, verbose=True)
        # self.training_sampler.calibration(training_loader, verbose=True)
        if self.checksequence:
            print("train_dataloader: this has been load again")
        return self.training_loader

    def val_dataloader(self):

        if self.checksequence:
            print("val_dataloader: this has been load again")
        # self.test_dataset = SemanticKittiDataset(self.config, set='validation',
        #                                     balance_classes=False)
        # self.test_sampler = SemanticKittiSampler(self.test_dataset)
        # test_loader = DataLoader(self.test_dataset,
        #                      batch_size=1,
        #                      sampler=self.test_sampler,
        #                      collate_fn=SemanticKittiCollate,
        #                      num_workers=self.config.input_threads,
        #                      pin_memory=True)
        # self.test_sampler.calib_max_in(self.config, test_loader, verbose=True)
        # self.test_sampler.calibration(test_loader, verbose=True)
        self.valadation_dataset = self.test_loader
        self.valadation_dataset.dataset.val_points = []
        self.valadation_dataset.dataset.val_labels = []
        self.nc_tot = self.valadation_dataset.dataset.num_classes

        return self.valadation_dataset

    def test_dataloader(self):
        # test_loader = DataLoader(self.test_dataset,
        #                      batch_size=1,
        #                      sampler=self.test_sampler,
        #                      collate_fn=SemanticKittiCollate,
        #                      num_workers=self.config.input_threads,
        #                      pin_memory=True)
        return self.test_loader

    def on_train_start(self):
        if self.checksequence:
            print("__________________on_train_start_____________________this should goes first")
        ################
        # Initialization
        ################

        if self.config.saving:
            # Training log file
            with open(join(self.config.saving_path, 'training.txt'), "w") as file:
                file.write('epochs steps out_loss offset_loss train_accuracy time\n')

            # Killing file (simply delete this file when you want to stop the training)
            self.PID_file = join(self.config.saving_path, 'running_PID.txt')
            if not exists(self.PID_file):
                with open(self.PID_file, "w") as file:
                    file.write('Launched with PyCharm')

            # Checkpoints directory
            self.checkpoint_directory = join(self.config.saving_path, 'checkpoints')
            if not exists(self.checkpoint_directory):
                makedirs(self.checkpoint_directory)
        else:
            self.checkpoint_directory = None
            self.PID_file = None

        # Loop variables
        t0 = time.time()
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)
        return None

    def on_train_epoch_start(self):
        if self.checksequence:
            print("__________________on_train_epoch_start_____________________this should go second\n")
        if self.current_epoch == self.config.max_epoch - 1 and exists(self.PID_file):
                remove(self.PID_file)
        return None

    def training_step(self, batch: dict, batch_idx: int):  # type: ignore
        if self.checksequence:
            print("__________________iteration: training_step_____________________this should start to iterate \n")
        outputs = self.net(batch, self.config)
        loss = self.net.loss(outputs, batch.labels)
        acc = self.net.accuracy(outputs, batch.labels)
        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_value_(self.net.parameters(), self.config.grad_clip_norm)
        
        self.log("overall_loss", loss)
        self.log("overall_acc", acc)

        if self.config.saving:
            with open(join(self.config.saving_path, 'training.txt'), "a") as file:
                message = '{:d} {:d} {:.3f} {:.3f} {:.3f}\n'
                file.write(message.format(self.current_epoch,
                                        batch_idx,
                                        self.net.output_loss,
                                        self.net.reg_loss,
                                        acc))
        return loss
    
    def on_train_epoch_end(self):
        if self.checksequence:
            print("__________________iteration: on_train_epoch_end_____________________this should go last \n")
        if self.current_epoch in self.config.lr_decays:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.config.lr_decays[self.current_epoch]

        if self.config.saving:
                # Get current state dict
                save_dict = {'epoch': self.current_epoch,
                            'model_state_dict': self.net.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'saving_path': self.config.saving_path}

                # Save current state of the network (for restoring purposes)
                checkpoint_path = join(self.checkpoint_directory, 'current_chkp.tar')
                torch.save(save_dict, checkpoint_path)

                # Save checkpoints occasionally
                if (self.current_epoch + 1) % self.config.checkpoint_gap == 0:
                    checkpoint_path = join(self.checkpoint_directory, 'chkp_{:04d}.tar'.format(self.current_epoch + 1))
                    torch.save(save_dict, checkpoint_path)
        return None

    def on_validation_start(self):
        if self.checksequence:
            print("__________________on_validation_start_____________________this should go each time before validation \n")
        if not exists (join(self.config.saving_path, 'val_preds')):
            makedirs(join(self.config.saving_path, 'val_preds'))

        # initiate the dataset validation containers
       
        # Number of classes including ignored labels

        #####################
        # Network predictions
        #####################

        self.predictions = []
        self.targets = []
        self.inds = []
        self.val_i = 0

        return None

    def validation_step(self, batch: dict, batch_idx: int):  # type: ignore  
        if self.checksequence:
            print("____validation_step____")
        val_smooth = 0.95
        self.softmax = torch.nn.Softmax(1)
        outputs = self.net(batch, self.config)
        # Get probs and labels
        
        stk_probs = self.softmax(outputs).cpu().detach().numpy()
        lengths = batch.lengths[0].cpu().numpy()
        f_inds = batch.frame_inds.cpu().numpy()
        r_inds_list = batch.reproj_inds
        r_mask_list = batch.reproj_masks
        labels_list = batch.val_labels
        # Get predictions and labels per instance
        # ***************************************

        i0 = 0
        for b_i, length in enumerate(lengths):

            # Get prediction
            probs = stk_probs[i0 : i0 + length]
            proj_inds = r_inds_list[b_i]
            proj_mask = r_mask_list[b_i]
            frame_labels = labels_list[b_i]
            s_ind = f_inds[b_i, 0]
            f_ind = f_inds[b_i, 1]

            # Project predictions on the frame points
            proj_probs = probs[proj_inds]

            # Safe check if only one point:
            if proj_probs.ndim < 2:
                proj_probs = np.expand_dims(proj_probs, 0)

            # Insert false columns for ignored labels
            for l_ind, label_value in enumerate(self.valadation_dataset.dataset.label_values):
                if label_value in self.valadation_dataset.dataset.ignored_labels:
                    proj_probs = np.insert(proj_probs, l_ind, 0, axis=1)

            # Predicted labels
            preds = self.valadation_dataset.dataset.label_values[np.argmax(proj_probs, axis=1)]

            # Save predictions in a binary file
            filename = '{:s}_{:07d}.npy'.format(self.valadation_dataset.dataset.sequences[s_ind], f_ind)
            filepath = join(self.config.saving_path, 'val_preds', filename)
            if exists(filepath):
                frame_preds = np.load(filepath)
            else:
                frame_preds = np.zeros(frame_labels.shape, dtype=np.uint8)
            frame_preds[proj_mask] = preds.astype(np.uint8)
            np.save(filepath, frame_preds)

            # Save some of the frame pots
            if f_ind % 20 == 0:
                seq_path = join(self.valadation_dataset.dataset.path, 'sequences', self.valadation_dataset.dataset.sequences[s_ind])
                velo_file = join(seq_path, 'velodyne', self.valadation_dataset.dataset.frames[s_ind][f_ind] + '.bin')
                frame_points = np.fromfile(velo_file, dtype=np.float32)
                frame_points = frame_points.reshape((-1, 4))
                write_ply(filepath[:-4] + '_pots.ply',
                            [frame_points[:, :3], frame_labels, frame_preds],
                            ['x', 'y', 'z', 'gt', 'pre'])

            # Update validation confusions
            frame_C = fast_confusion(frame_labels,
                                        frame_preds.astype(np.int32),
                                        self.valadation_dataset.dataset.label_values)
            self.valadation_dataset.dataset.val_confs[s_ind][f_ind, :, :] = frame_C

            # Stack all prediction for this epoch
            self.predictions += [preds]
            self.targets += [frame_labels[proj_mask]]
            self.inds += [f_inds[b_i, :]]
            self.val_i += 1
            i0 += length

        return 0

    def validation_epoch_end(self, outputs):
        if self.checksequence:
            print("__________________validation_epoch_end_____________________this should go at the end of validation \n")
        # Confusions for our subparts of validation set
        Confs = np.zeros((len(self.predictions), self.nc_tot, self.nc_tot), dtype=np.int32)
        for i, (preds, truth) in enumerate(zip(self.predictions, self.targets)):

            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, self.valadation_dataset.dataset.label_values).astype(np.int32)

        t3 = time.time()

        #######################################
        # Results on this subpart of validation
        #######################################

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Balance with real validation proportions
        C *= np.expand_dims(self.valadation_dataset.dataset.class_proportions / (np.sum(C, axis=1) + 1e-6), 1)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(self.valadation_dataset.dataset.label_values))):
            if label_value in self.valadation_dataset.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Objects IoU
        IoUs = IoU_from_confusions(C)

        #####################################
        # Results on the whole validation set
        #####################################

        t4 = time.time()

        # Sum all validation confusions
        C_tot = [np.sum(seq_C, axis=0) for seq_C in self.valadation_dataset.dataset.val_confs if len(seq_C) > 0]
        C_tot = np.sum(np.stack(C_tot, axis=0), axis=0)

        if self.debug:
            s = '\n'
            for cc in C_tot:
                for c in cc:
                    s += '{:8.1f} '.format(c)
                s += '\n'
            print(s)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(self.valadation_dataset.dataset.label_values))):
            if label_value in self.valadation_dataset.dataset.ignored_labels:
                C_tot = np.delete(C_tot, l_ind, axis=0)
                C_tot = np.delete(C_tot, l_ind, axis=1)

        # Objects IoU
        val_IoUs = IoU_from_confusions(C_tot)

        t5 = time.time()

        # Saving (optionnal)
        if self.config.saving:

            IoU_list = [IoUs, val_IoUs]
            file_list = ['subpart_IoUs.txt', 'val_IoUs.txt']
            for IoUs_to_save, IoU_file in zip(IoU_list, file_list):

                # Name of saving file
                test_file = join(self.config.saving_path, IoU_file)

                # Line to write:
                line = ''
                for IoU in IoUs_to_save:
                    line += '{:.3f} '.format(IoU)
                line = line + '\n'

                # Write in file
                if exists(test_file):
                    with open(test_file, "a") as text_file:
                        text_file.write(line)
                else:
                    with open(test_file, "w") as text_file:
                        text_file.write(line)

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        print('{:s} : subpart mIoU = {:.1f} %'.format(self.config.dataset, mIoU))
        mIoU = 100 * np.mean(val_IoUs)
        print('{:s} :     val mIoU = {:.1f} %'.format(self.config.dataset, mIoU))
        return 0

    def test_step(self, batch: dict, batch_idx: int):  # type: ignore
        return None

    def configure_optimizers(self) -> torch.optim.Adam:
        deform_params = [v for k, v in self.net.named_parameters() if 'offset' in k]
        other_params = [v for k, v in self.net.named_parameters() if 'offset' not in k]
        deform_lr = self.config.learning_rate * self.config.deform_lr_factor
        self.optimizer = torch.optim.SGD([{'params': other_params},
                                          {'params': deform_params, 'lr': deform_lr}],
                                         lr=self.config.learning_rate,
                                         momentum=self.config.momentum,
                                         weight_decay=self.config.weight_decay)
        return self.optimizer
