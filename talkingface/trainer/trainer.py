import os

from logging import getLogger
from time import time
import json, subprocess
import torch.nn.functional as F
import glob
import itertools
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
import torch.cuda.amp as amp
from torch import nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import re
import yaml
import sys
sys.path.append('saved/stargan/pwg')
from parallel_wavegan.utils import load_model
from parallel_wavegan.utils import read_hdf5
import soundfile as sf
import librosa

from talkingface.utils import(
    ensure_dir,
    get_local_time,
    early_stopping,
    calculate_valid_score,
    dict2str,
    get_tensorboard,
    set_color,
    get_gpu_usage,
    WandbLogger
)
from talkingface.data.dataprocess.wav2lip_process import Wav2LipAudio
from talkingface.evaluator import Evaluator


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model
    
    def fit(self, train_data):
        r"""Train the model based on the train data."""
        raise NotImplementedError("Method [next] should be implemented.")

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data."""

        raise NotImplementedError("Method [next] should be implemented.")
    

class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in talkingface systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
    resume_checkpoint() and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    `model` is the instantiated object of a Model Class.

    """
    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)
        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.wandblogger = WandbLogger(config)
        # self.enable_amp = config["enable_amp"]
        # self.enable_scaler = torch.cuda.is_available() and config["enable_scaler"]

        # config for train 
        self.learner = config["learner"]
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.eval_step = min(config["eval_step"], self.epochs)
        self.stopping_step = config["stopping_step"]
        self.test_batch_size = config["eval_batch_size"]
        self.gpu_available = torch.cuda.is_available() and config["use_gpu"]
        self.device = config["device"]
        self.checkpoint_dir = config["checkpoint_dir"]
        ensure_dir(self.checkpoint_dir)
        saved_model_file = "{}-{}.pth".format(self.config["model"], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.weight_decay = config["weight_decay"]
        self.start_epoch = 0
        self.cur_step = 0
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.evaluator = Evaluator(config)

        self.valid_metric_bigger = config["valid_metric_bigger"]
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None

    def _build_optimizer(self, **kwargs):
        params = kwargs.pop("params", self.model.parameters())
        learner = kwargs.pop("learner", self.learner)
        learning_rate = kwargs.pop("learning_rate", self.learning_rate)
        weight_decay = kwargs.pop("weight_decay", self.weight_decay)
        if (self.config["reg_weight"] and weight_decay and weight_decay * self.config["reg_weight"] > 0):
            self.logger.warning(
                "The parameters [weight_decay] and [reg_weight] are specified simultaneously, "
                "which may lead to double regularization."
            )

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adamw":
            optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "sparse_adam":
            optimizer = optim.SparseAdam(params, lr=learning_rate)
            if weight_decay > 0:
                self.logger.warning(
                    "Sparse Adam cannot argument received argument [{weight_decay}]"
                )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            the averaged loss of this epoch
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss_dict = {}
        step = 0
        iter_data = (
            tqdm(
            train_data,
            total=len(train_data),
            ncols=None,
            )
            if show_progress
            else train_data
        )

        for batch_idx, interaction in enumerate(iter_data):
            self.optimizer.zero_grad()
            step += 1
            losses_dict = loss_func(interaction)
            loss = losses_dict["loss"]

            for key, value in losses_dict.items():
                if key in total_loss_dict:
                    if not torch.is_tensor(value):
                        total_loss_dict[key] += value
                    # 如果键已经在总和字典中，累加当前值
                    else:
                        losses_dict[key] = value.item()
                        total_loss_dict[key] += value.item()
                else:
                    if not torch.is_tensor(value):
                        total_loss_dict[key] = value
                    # 否则，将当前值添加到字典中
                    else:
                        losses_dict[key] = value.item()
                        total_loss_dict[key] = value.item()
            iter_data.set_description(set_color(f"train {epoch_idx} {losses_dict}", "pink"))

            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
        average_loss_dict = {}
        for key, value in total_loss_dict.items():
            average_loss_dict[key] = value/step

        return average_loss_dict

        

    def _valid_epoch(self, valid_data, show_progress=False):
        r"""Valid the model with valid data. Different from the evaluate, this is use for training.

        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            loss
        """
        print('Valid for {} steps'.format(self.eval_steps))
        self.model.eval()
        total_loss_dict = {}
        iter_data = (
            tqdm(valid_data,
                total=len(valid_data),
                ncols=None,
            )
            if show_progress
            else valid_data
        )
        step = 0
        for batch_idx, batched_data in enumerate(iter_data):
            step += 1
            batched_data.to(self.device)    
            losses_dict = self.model.calculate_loss(batched_data, valid=True)
            for key, value in losses_dict.items():
                if key in total_loss_dict:
                    if not torch.is_tensor(value):
                        total_loss_dict[key] += value
                    # 如果键已经在总和字典中，累加当前值
                    else:
                        losses_dict[key] = value.item()
                        total_loss_dict[key] += value.item()
                else:
                    if not torch.is_tensor(value):
                        total_loss_dict[key] = value
                    # 否则，将当前值添加到字典中
                    else:
                        losses_dict[key] = value.item()
                        total_loss_dict[key] = value.item()
            iter_data.set_description(set_color(f"Valid {losses_dict}", "pink"))
        average_loss_dict = {}
        for key, value in total_loss_dict.items():
            average_loss_dict[key] = value/step

        return average_loss_dict
                


    
    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        saved_model_file = kwargs.pop("saved_model_file", self.saved_model_file)
        state = {
            "config": self.config,
            "epoch": epoch,
            "cur_step": self.cur_step,
            "best_valid_score": self.best_valid_score,
            "state_dict": self.model.state_dict(),
            "other_parameter": self.model.other_parameter(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file, pickle_protocol=4)
        if verbose:
            self.logger.info(
                set_color("Saving current", "blue") + f": {saved_model_file}"
            )

    def resume_checkpoint(self, resume_file):
        r"""Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file

        """
        resume_file = str(resume_file)
        self.saved_model_file = resume_file
        checkpoint = torch.load(resume_file, map_location=self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.cur_step = checkpoint["cur_step"]
        # self.best_valid_score = checkpoint["best_valid_score"]

        # load architecture params from checkpoint
        if checkpoint["config"]["model"].lower() != self.config["model"].lower():
            self.logger.warning(
                "Architecture configuration given in config file is different from that of checkpoint. "
                "This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.load_other_parameter(checkpoint.get("other_parameter"))

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        message_output = "Checkpoint loaded. Resume training from epoch {}".format(
            self.start_epoch
        )
        self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = self.config["loss_decimal_place"] or 4
        train_loss_output = (
            set_color(f"epoch {epoch_idx} training", "green")
            + " ["
            + set_color("time", "blue")
            + f": {e_time - s_time:.2f}s, "
        )
        # 遍历字典，格式化并添加每个损失项
        loss_items = [
            set_color(f"{key}", "blue") + f": {value:.{des}f}"
            for key, value in losses.items()
        ]
        # 将所有损失项连接成一个字符串，并与前面的输出拼接
        train_loss_output += ", ".join(loss_items)
        return train_loss_output + "]"

    def _add_hparam_to_tensorboard(self, best_valid_result):
        # base hparam
        hparam_dict = {
            "learner": self.config["learner"],
            "learning_rate": self.config["learning_rate"],
            "train_batch_size": self.config["train_batch_size"],
        }
        # unrecorded parameter
        unrecorded_parameter = {
            parameter
            for parameters in self.config.parameters.values()
            for parameter in parameters
        }.union({"model", "dataset", "config_files", "device"})
        # other model-specific hparam
        hparam_dict.update(
            {
                para: val
                for para, val in self.config.final_config_dict.items()
                if para not in unrecorded_parameter
            }
        )
        for k in hparam_dict:
            if hparam_dict[k] is not None and not isinstance(
                hparam_dict[k], (bool, str, float, int)
            ):
                hparam_dict[k] = str(hparam_dict[k])

        self.tensorboard.add_hparams(
            hparam_dict, {"hparam/best_valid_result": best_valid_result}
        )

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                            If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
            best result
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        if not (self.config['resume_checkpoint_path'] == None ) and self.config['resume']:
            self.resume_checkpoint(self.config['resume_checkpoint_path'])
        
        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss)
            
            if verbose:
                self.logger.info(train_loss_output)
            # self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_loss = self._valid_epoch(valid_data=valid_data, show_progress=show_progress)

                (self.best_valid_score, self.cur_step, stop_flag,update_flag,) = early_stopping(
                    valid_loss['loss'],
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()

                valid_loss_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(valid_loss)
                )
                if verbose:
                    self.logger.info(valid_loss_output)

                
                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_loss['loss']

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break
    @torch.no_grad()
    def evaluate(self, load_best_model=True, model_file=None):
        """
        Evaluate the model based on the test data.

        args: load_best_model: bool, whether to load the best model in the training process.
                model_file: str, the model file you want to evaluate.

        """
        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = "Loading model structure and parameters from {}".format(
                checkpoint_file
            )
            self.logger.info(message_output)
        self.model.eval()

        datadict = self.model.generate_batch()
        eval_result = self.evaluator.evaluate(datadict)
        self.logger.info(eval_result)



class Wav2LipTrainer(Trainer):
    def __init__(self, config, model):
        super(Wav2LipTrainer, self).__init__(config, model)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            the averaged loss of this epoch
        """
        self.model.train()



        loss_func = loss_func or self.model.calculate_loss
        total_loss_dict = {}
        step = 0
        iter_data = (
            tqdm(
            train_data,
            total=len(train_data),
            ncols=None,
            )
            if show_progress
            else train_data
        )

        for batch_idx, interaction in enumerate(iter_data):
            self.optimizer.zero_grad()
            step += 1
            losses_dict = loss_func(interaction)
            loss = losses_dict["loss"]

            for key, value in losses_dict.items():
                if key in total_loss_dict:
                    if not torch.is_tensor(value):
                        total_loss_dict[key] += value
                    # 如果键已经在总和字典中，累加当前值
                    else:
                        losses_dict[key] = value.item()
                        total_loss_dict[key] += value.item()
                else:
                    if not torch.is_tensor(value):
                        total_loss_dict[key] = value
                    # 否则，将当前值添加到字典中
                    else:
                        losses_dict[key] = value.item()
                        total_loss_dict[key] = value.item()
            iter_data.set_description(set_color(f"train {epoch_idx} {losses_dict}", "pink"))

            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
        average_loss_dict = {}
        for key, value in total_loss_dict.items():
            average_loss_dict[key] = value/step

        return average_loss_dict

        
    
    def _valid_epoch(self, valid_data, loss_func=None, show_progress=False):
        print('Valid'.format(self.eval_step))
        self.model.eval()
        total_loss_dict = {}
        iter_data = (
            tqdm(valid_data,
                total=len(valid_data),
                ncols=None,
                desc=set_color("Valid", "pink")
            )
            if show_progress
            else valid_data
        )
        step = 0
        for batch_idx, batched_data in enumerate(iter_data):
            step += 1
            losses_dict = self.model.calculate_loss(batched_data, valid=True)
            for key, value in losses_dict.items():
                if key in total_loss_dict:
                    if not torch.is_tensor(value):
                        total_loss_dict[key] += value
                    # 如果键已经在总和字典中，累加当前值
                    else:
                        losses_dict[key] = value.item()
                        total_loss_dict[key] += value.item()
                else:
                    if not torch.is_tensor(value):
                        total_loss_dict[key] = value
                    # 否则，将当前值添加到字典中
                    else:
                        losses_dict[key] = value.item()
                        total_loss_dict[key] = value.item()
        average_loss_dict = {}
        for key, value in total_loss_dict.items():
            average_loss_dict[key] = value/step
        if losses_dict["sync_loss"] < .75:
            self.model.config["syncnet_wt"] = 0.01
        return average_loss_dict

class starganTrainer(Trainer):
    def __init__(self, config, model):
        super(starganTrainer, self).__init__(config, model)
        self.config = config
        
    def comb(self, N, r):
        iterable = list(range(0,N))
        return list(itertools.combinations(iterable,2))

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            the averaged loss of this epoch
        """
        w_adv = self.config['w_adv']
        w_grad = self.config['w_grad']
        w_cls = self.config['w_cls']
        w_cyc = self.config['w_cyc']
        w_rec = self.config['w_rec']
        lrate_g = self.config['lrate_g']
        lrate_d = self.config['lrate_d']
        gradient_clip = self.config['gradient_clip']
        device = self.config['device']
        
        optimizers = {
            'gen' : optim.Adam(self.model.gen.parameters(), lr=lrate_g, betas=(0.9,0.999)),
            'dis' : optim.Adam(self.model.dis.parameters(), lr=lrate_d, betas=(0.5,0.999))
        }

        for X_list in train_data:
            n_spk = len(X_list)
            xin = []
            for s in range(n_spk):
                xin.append(torch.tensor(X_list[s]).to(device, dtype=torch.float))

            # List of speaker pairs
            spk_pair_list = self.comb(n_spk,2)
            n_spk_pair = len(spk_pair_list)

            gen_loss_mean = 0
            dis_loss_mean = 0
            advloss_d_mean = 0
            gradloss_d_mean = 0
            advloss_g_mean = 0
            clsloss_d_mean = 0
            clsloss_g_mean = 0
            cycloss_mean = 0
            recloss_mean = 0
            # Iterate through all speaker pairs
            for m in range(n_spk_pair):
                s0 = spk_pair_list[m][0]
                s1 = spk_pair_list[m][1]

                AdvLoss_g, ClsLoss_g, CycLoss, RecLoss = self.model.calc_gen_loss(xin[s0], xin[s1], s0, s1)
                gen_loss = (w_adv * AdvLoss_g + w_cls * ClsLoss_g + w_cyc * CycLoss + w_rec * RecLoss)

                self.model.gen.zero_grad()
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.gen.parameters(), gradient_clip)
                optimizers['gen'].step()
                
                AdvLoss_d, GradLoss_d, ClsLoss_d = self.model.calc_dis_loss(xin[s0], xin[s1], s0, s1)
                dis_loss = w_adv * AdvLoss_d + w_grad * GradLoss_d + w_cls * ClsLoss_d

                self.model.dis.zero_grad()
                dis_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.dis.parameters(), gradient_clip)
                optimizers['dis'].step()

                gen_loss_mean += gen_loss.item()
                dis_loss_mean += dis_loss.item()
                advloss_d_mean += AdvLoss_d.item()
                gradloss_d_mean += GradLoss_d.item()
                advloss_g_mean += AdvLoss_g.item()
                clsloss_d_mean += ClsLoss_d.item()
                clsloss_g_mean += ClsLoss_g.item()
                cycloss_mean += CycLoss.item()
                recloss_mean += RecLoss.item()

            gen_loss_mean /= n_spk_pair
            dis_loss_mean /= n_spk_pair
            advloss_d_mean /= n_spk_pair
            gradloss_d_mean /= n_spk_pair
            advloss_g_mean /= n_spk_pair
            clsloss_d_mean /= n_spk_pair
            clsloss_g_mean /= n_spk_pair
            cycloss_mean /= n_spk_pair
            recloss_mean /= n_spk_pair
            
            return {'gen_loss_mean': gen_loss_mean, 'dis_loss_mean': dis_loss_mean, 'advloss_d_mean': advloss_d_mean, 'gradloss_d_mean': gradloss_d_mean, 'advloss_g_mean': advloss_g_mean, 'clsloss_d_mean': clsloss_d_mean, 'clsloss_g_mean': clsloss_g_mean, 'cycloss_mean': cycloss_mean, 'recloss_mean': recloss_mean}
        

    def _valid_epoch(self, valid_data, show_progress=False):
        r"""Valid the model with valid data. Different from the evaluate, this is use for training.

        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            loss
        """
        w_adv = self.config['w_adv']
        w_grad = self.config['w_grad']
        w_cls = self.config['w_cls']
        w_cyc = self.config['w_cyc']
        w_rec = self.config['w_rec']
        lrate_g = self.config['lrate_g']
        lrate_d = self.config['lrate_d']
        gradient_clip = self.config['gradient_clip']
        device = self.config['device']
        
        optimizers = {
            'gen' : optim.Adam(self.model.gen.parameters(), lr=lrate_g, betas=(0.9,0.999)),
            'dis' : optim.Adam(self.model.dis.parameters(), lr=lrate_d, betas=(0.5,0.999))
        }

        for X_list in valid_data:
            n_spk = len(X_list)
            xin = []
            for s in range(n_spk):
                xin.append(torch.tensor(X_list[s]).to(device, dtype=torch.float))

            # List of speaker pairs
            spk_pair_list = self.comb(n_spk,2)
            n_spk_pair = len(spk_pair_list)

            gen_loss_mean = 0
            dis_loss_mean = 0
            advloss_d_mean = 0
            gradloss_d_mean = 0
            advloss_g_mean = 0
            clsloss_d_mean = 0
            clsloss_g_mean = 0
            cycloss_mean = 0
            recloss_mean = 0
            # Iterate through all speaker pairs
            for m in range(n_spk_pair):
                s0 = spk_pair_list[m][0]
                s1 = spk_pair_list[m][1]

                AdvLoss_g, ClsLoss_g, CycLoss, RecLoss = self.model.calc_gen_loss(xin[s0], xin[s1], s0, s1)
                gen_loss = (w_adv * AdvLoss_g + w_cls * ClsLoss_g + w_cyc * CycLoss + w_rec * RecLoss)

                self.model.gen.zero_grad()
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.gen.parameters(), gradient_clip)
                optimizers['gen'].step()
                
                AdvLoss_d, GradLoss_d, ClsLoss_d = self.model.calc_dis_loss(xin[s0], xin[s1], s0, s1)
                dis_loss = w_adv * AdvLoss_d + w_grad * GradLoss_d + w_cls * ClsLoss_d

                self.model.dis.zero_grad()
                dis_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.dis.parameters(), gradient_clip)
                optimizers['dis'].step()

                gen_loss_mean += gen_loss.item()
                dis_loss_mean += dis_loss.item()
                advloss_d_mean += AdvLoss_d.item()
                gradloss_d_mean += GradLoss_d.item()
                advloss_g_mean += AdvLoss_g.item()
                clsloss_d_mean += ClsLoss_d.item()
                clsloss_g_mean += ClsLoss_g.item()
                cycloss_mean += CycLoss.item()
                recloss_mean += RecLoss.item()

            gen_loss_mean /= n_spk_pair
            dis_loss_mean /= n_spk_pair
            advloss_d_mean /= n_spk_pair
            gradloss_d_mean /= n_spk_pair
            advloss_g_mean /= n_spk_pair
            clsloss_d_mean /= n_spk_pair
            clsloss_g_mean /= n_spk_pair
            cycloss_mean /= n_spk_pair
            recloss_mean /= n_spk_pair
            
            return {'loss': gen_loss_mean+dis_loss_mean, 'gen_loss_mean': gen_loss_mean, 'dis_loss_mean': dis_loss_mean, 'advloss_d_mean': advloss_d_mean, 'gradloss_d_mean': gradloss_d_mean, 'advloss_g_mean': advloss_g_mean, 'clsloss_d_mean': clsloss_d_mean, 'clsloss_g_mean': clsloss_g_mean, 'cycloss_mean': cycloss_mean, 'recloss_mean': recloss_mean}
        
    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                            If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
            best result
        """
        lrate_g = self.config['lrate_g']
        lrate_d = self.config['lrate_d']
        resume = self.config['resume']
        snapshot = self.config['snapshot']
        model_dir = os.path.join(self.config['model_rootdir'], self.config['experiment_name'])
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        optimizers = {
            'gen' : optim.Adam(self.model.gen.parameters(), lr=lrate_g, betas=(0.9,0.999)),
            'dis' : optim.Adam(self.model.dis.parameters(), lr=lrate_d, betas=(0.5,0.999))
        }

        for tag in ['gen', 'dis']:
            checkpointpath = os.path.join(model_dir, '{}.{}.pt'.format(resume,tag))
            if os.path.exists(checkpointpath):
                checkpoint = torch.load(checkpointpath, map_location=device)
                if tag == 'gen':
                    self.model.gen.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.dis.load_state_dict(checkpoint['model_state_dict'])
                optimizers[tag].load_state_dict(checkpoint['optimizer_state_dict'])
                print('{} loaded successfully.'.format(checkpointpath))
        
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        if not (self.config['resume_checkpoint_path'] == None ) and self.config['resume']:
            self.resume_checkpoint(self.config['resume_checkpoint_path'])
        
        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss)
            
            if epoch_idx % snapshot == 0:
                for tag in ['gen', 'dis']:
                    print('save {} at {} epoch'.format(tag, epoch_idx))
                    if tag == 'gen':
                        torch.save({'epoch': epoch_idx,
                                    'model_state_dict': self.model.gen.state_dict(),
                                    'optimizer_state_dict': optimizers[tag].state_dict()},
                                    os.path.join(model_dir, '{}.{}.pt'.format(epoch_idx, tag)))
                    else:
                        torch.save({'epoch': epoch_idx,
                                    'model_state_dict': self.model.dis.state_dict(),
                                    'optimizer_state_dict': optimizers[tag].state_dict()},
                                    os.path.join(model_dir, '{}.{}.pt'.format(epoch_idx, tag)))
            
            if verbose:
                self.logger.info(train_loss_output)
            # self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_loss = self._valid_epoch(valid_data=valid_data, show_progress=show_progress)
                valid_end_time = time()

                valid_loss_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(valid_loss)
                )
                if verbose:
                    self.logger.info(valid_loss_output)

    @torch.no_grad()
    def evaluate(self, load_best_model=True, model_file=None):
        """
        Evaluate the model based on the test data.

        args: load_best_model: bool, whether to load the best model in the training process.
                model_file: str, the model file you want to evaluate.

        """
        num_mels = self.config['num_mels']
        arch_type = self.config['arch_type']
        loss_type = self.config['loss_type']
        n_spk = self.config['n_spk']
        trg_spk_list = self.config['spk_list']
        zdim = self.config['zdim']
        hdim = self.config['hdim']
        mdim = self.config['mdim']
        sdim = self.config['sdim']
        normtype = self.config['normtype']
        src_conditioning = self.config['src_conditioning']
        model_dir = os.path.join(self.config['model_rootdir'], self.config['experiment_name'])
        checkpoint = self.config['checkpoint']
        device = self.config['device']
        input_dir = self.config['src3']
        dataset_yaml_path = self.config['dataset_yaml_path']
        model_yaml_path = self.config['model_yaml_path']
        out = self.config['out']
        experiment_name = self.config['experiment_name']

        stat_filepath = self.config['stat']
        melspec_scaler = StandardScaler()
        if os.path.exists(stat_filepath):
            with open(stat_filepath, mode='rb') as f:
                melspec_scaler = pickle.load(f)
            print('Loaded mel-spectrogram statistics successfully.')
        else:
            print('Stat file not found.')

        for tag in ['gen', 'dis']:
            vc_checkpoint_idx = self.find_newest_model_file(model_dir, tag) if checkpoint <= 0 else checkpoint
            mfilename = '{}.{}.pt'.format(vc_checkpoint_idx,tag)
            path = os.path.join(self.config['model_rootdir'],self.config['experiment_name'],mfilename)
            if path is not None:
                model_checkpoint = torch.load(path, map_location=device)
                if tag == 'gen':
                    self.model.gen.load_state_dict(model_checkpoint['model_state_dict'])
                else:
                    self.model.dis.load_state_dict(model_checkpoint['model_state_dict'])
                print('{}: {}'.format(tag, os.path.abspath(path)))

        for tag in ['gen', 'dis']:
            if tag == 'gen':
                self.model.gen.to(device).train(mode=True)
            else:
                self.model.dis.to(device).train(mode=True)
                
        # Set up nv
        vocoder = self.config['vocoder']
        voc_dir = self.config['voc_dir']
        voc_yaml_path = os.path.join(voc_dir,'conf', '{}.yaml'.format(vocoder))
        checkpointlist = self.listdir_ext(
            os.path.join(voc_dir,'exp','train_nodev_all_{}'.format(vocoder)),'.pkl')
        nv_checkpoint = os.path.join(voc_dir,'exp',
                                      'train_nodev_all_{}'.format(vocoder),
                                      checkpointlist[-1]) # Find and use the newest checkpoint model.
        print('vocoder: {}'.format(os.path.abspath(nv_checkpoint)))

        with open(voc_yaml_path) as f:
            nv_config = yaml.load(f, Loader=yaml.Loader)
            
        with open(dataset_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        nv_config.update(config)
        with open(model_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        nv_config.update(config)

        model_nv = load_model(nv_checkpoint, nv_config)
        model_nv.remove_weight_norm()
        model_nv = model_nv.eval().to(device)

        src_spk_list = sorted(os.listdir(input_dir))
        
        for i, src_spk in enumerate(src_spk_list):
            src_wav_dir = os.path.join(input_dir, src_spk)
            for j, trg_spk in enumerate(trg_spk_list):
                if src_spk != trg_spk:
                    print('Converting {}2{}...'.format(src_spk, trg_spk))
                    for n, src_wav_filename in enumerate(os.listdir(src_wav_dir)):
                        src_wav_filepath = os.path.join(src_wav_dir, src_wav_filename)
                        src_melspec = self.audio_transform(src_wav_filepath, melspec_scaler, device)
                        k_t = j
                        k_s = i if src_conditioning else None

                        conv_melspec = self.model(src_melspec, k_t, k_s)

                        conv_melspec = conv_melspec[0,:,:].detach().cpu().clone().numpy()
                        conv_melspec = conv_melspec.T # n_frames x n_mels

                        out_wavpath = os.path.join(out,experiment_name,'{}'.format(vc_checkpoint_idx),vocoder,'{}2{}'.format(src_spk,trg_spk), src_wav_filename)
                        self.synthesis(conv_melspec, model_nv, nv_config, out_wavpath, device)
        
    def find_newest_model_file(self, model_dir, tag):
        mfile_list = os.listdir(model_dir)
        checkpoint = max([int(os.path.splitext(os.path.splitext(mfile)[0])[0]) for mfile in mfile_list if mfile.endswith('.{}.pt'.format(tag))])
        return checkpoint
    
    def extract_num(self, s, p, ret=0):
        search = p.search(s)
        if search:
            return int(search.groups()[0])
        else:
            return ret

    def listdir_ext(self, dirpath, ext):
        p = re.compile(r'(\d+)')
        out = []
        for file in sorted(os.listdir(dirpath), key=lambda s: self.extract_num(s, p)):
            if os.path.splitext(file)[1]==ext:
                out.append(file)
        return out
    
    def audio_transform(self, wav_filepath, scaler, device):

        trim_silence = self.config['trim_silence']
        top_db = self.config['top_db']
        flen = self.config['flen']
        fshift = self.config['fshift']
        fmin = self.config['fmin']
        fmax = self.config['fmax']
        num_mels = self.config['num_mels']
        fs = self.config['fs']

        audio, fs_ = sf.read(wav_filepath)
        if trim_silence:
            #print('trimming.')
            audio, _ = librosa.effects.trim(audio, top_db=top_db, frame_length=2048, hop_length=512)
        if fs != fs_:
            #print('resampling.')
            audio = librosa.resample(audio, fs_, fs)
        melspec_raw = self.logmelfilterbank(audio,fs, fft_size=flen,hop_size=fshift,
                                        fmin=fmin, fmax=fmax, num_mels=num_mels)
        melspec_raw = melspec_raw.astype(np.float32) # n_frame x n_mels

        melspec_norm = scaler.transform(melspec_raw)
        melspec_norm =  melspec_norm.T # n_mels x n_frame

        return torch.tensor(melspec_norm[None]).to(device, dtype=torch.float)
    
    def logmelfilterbank(self,
                         audio,
                         sampling_rate,
                         fft_size=1024,
                         hop_size=256,
                         win_length=None,
                         window="hann",
                         num_mels=80,
                         fmin=None,
                         fmax=None,
                         eps=1e-10,
                         ):
        """Compute log-Mel filterbank feature.
        Args:
            audio (ndarray): Audio signal (T,).
            sampling_rate (int): Sampling rate.
            fft_size (int): FFT size.
            hop_size (int): Hop size.
            win_length (int): Window length. If set to None, it will be the same as fft_size.
            window (str): Window function type.
            num_mels (int): Number of mel basis.
            fmin (int): Minimum frequency in mel basis calculation.
            fmax (int): Maximum frequency in mel basis calculation.
            eps (float): Epsilon value to avoid inf in log calculation.
        Returns:
            ndarray: Log Mel filterbank feature (#frames, num_mels).
        """
        # get amplitude spectrogram
        x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                              win_length=win_length, window=window, pad_mode="reflect")
        spc = np.abs(x_stft).T  # (#frames, #bins)

        # get mel basis
        fmin = 0 if fmin is None else fmin
        fmax = sampling_rate / 2 if fmax is None else fmax
        mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)

        return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))
    
    def synthesis(self, melspec, model_nv, nv_config, savepath, device):
        ## Parallel WaveGAN / MelGAN
        melspec = torch.tensor(melspec, dtype=torch.float).to(device)
        #start = time.time()
        x = model_nv.inference(melspec).view(-1)
        #elapsed_time = time.time() - start
        #rtf2 = elapsed_time/audio_len
        #print ("elapsed_time (waveform generation): {0}".format(elapsed_time) + "[sec]")
        #print ("real time factor (waveform generation): {0}".format(rtf2))

        # save as PCM 16 bit wav file
        if not os.path.exists(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))
        sf.write(savepath, x.detach().cpu().clone().numpy(), nv_config["sampling_rate"], "PCM_16")