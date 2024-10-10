import torch
import os
import numpy as np
import random
from tensorboardX import SummaryWriter
from einops import repeat
from contextlib import contextmanager
import time
import yacs
from yacs.config import CfgNode as CN
import wandb


def seed_np_torch(seed=20001118):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger():
    def __init__(self, path) -> None:
        self.writer = SummaryWriter(logdir=path, flush_secs=1)
        self.tag_step = {}

    def log(self, tag, value):
        if tag not in self.tag_step:
            self.tag_step[tag] = 0
        else:
            self.tag_step[tag] += 1
        if "video" in tag:
            self.writer.add_video(tag, value, self.tag_step[tag], fps=15)
        elif "images" in tag:
            self.writer.add_images(tag, value, self.tag_step[tag])
        elif "hist" in tag:
            self.writer.add_histogram(tag, value, self.tag_step[tag])
        else:
            self.writer.add_scalar(tag, value, self.tag_step[tag])



class WandbLogger():
    def __init__(self, config, project=None, mode='online'):
        """
        Initialize the Logger class.

        Args:
            path (str): Path to the directory where logs will be saved. This can be used to define the run name in W&B.
            project (str, optional): Name of the W&B project. Defaults to None.
        """
        # Initialize a W&B run with the given project and path as the run name
        pure_env_name = config.BasicSettings.Env_name.split('/')[-1].split('-')[0]
        run_name = f"{config.Models.WorldModel.Backbone}_{config.Models.Agent.Policy}_{pure_env_name}_seed{config.BasicSettings.Seed}"
        self.run = wandb.init(project=project, config=config, mode=mode, name=run_name)
        self.run.name = f"{self.run.name}_{self.run.id}"
        self.tag_step = {}


    def log(self, tag, value, global_step):
        """
        Log data to Weights & Biases.

        Args:
            tag (str): The tag or label for the data being logged.
            value: The data to be logged. It can be a scalar, image, histogram, or video.
        """
        # Log data based on the type
        if "video" in tag:
            # Log video
            wandb.log({tag: wandb.Video(value, fps=1, format='gif')}, step=global_step)
        elif "images" in tag:
            # Log images
            images = [wandb.Image(img) for img in value]  # Convert each image to a wandb.Image
            wandb.log({tag: images}, step=global_step)
        elif "hist" in tag:
            # Log histogram
            wandb.log({tag: wandb.Histogram(value)}, step=global_step)
        else:
            # Log scalar value
            wandb.log({tag: value}, step=global_step)

    def update_config(self, update_dict):
        """
        Update the configuration with the given parameters.

        Args:
            update_dict (dict): A dictionary containing scalar parameter information to update in the configuration.
        """
        # Update the configuration using wandb.config.update
        wandb.config.update(update_dict)

    def close(self):
        """
        Finalize and close the W&B run.
        """
        # Finish the run
        wandb.finish()



class EMAScalar():
    def __init__(self, decay) -> None:
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar


def load_config(args):
    conf = CN()

    conf.BasicSettings = CN()
    conf.BasicSettings.Seed = args.seed
    conf.BasicSettings.Env_name=args.env_name
    conf.BasicSettings.Device = args.device
    conf.BasicSettings.Model = args.model
    conf.BasicSettings.ImageSize = 0
    conf.BasicSettings.ReplayBufferOnGPU = False
    conf.n = args.n # note

    # Under this setting, input 128*128 -> latent 16*16*64
    conf.Models = CN()

    conf.Models.WorldModel = CN()
    conf.Models.WorldModel.InChannels = 0
    conf.Models.WorldModel.TransformerMaxLength = 0
    conf.Models.WorldModel.TransformerHiddenDim = 0
    conf.Models.WorldModel.TransformerNumLayers = 0
    conf.Models.WorldModel.TransformerNumHeads = 0

    conf.Models.WorldModel.Act = 'ReLU'

    conf.Models.WorldModel.Mamba_d_model = 0
    conf.Models.WorldModel.Mamba_n_layer = 0
    conf.Models.WorldModel.Mamba_d_state = 0


    conf.Models.Agent = CN()
    conf.Models.Agent.AC.NumLayers = 0
    conf.Models.Agent.AC.HiddenDim = 256
    conf.Models.Agent.AC.Gamma = 1.0
    conf.Models.Agent.AC.Lambda = 0.0
    conf.Models.Agent.AC.EntropyCoef = 0.0
    conf.Models.Agent.AC.Act = 'ReLU'


    conf.JointTrainAgent = CN()
    conf.JointTrainAgent.SampleMaxSteps = 0
    conf.JointTrainAgent.BufferMaxLength = 0
    conf.JointTrainAgent.BufferWarmUp = 0
    conf.JointTrainAgent.NumEnvs = 0
    conf.JointTrainAgent.BatchSize = 0
    conf.JointTrainAgent.BatchLength = 0
    conf.JointTrainAgent.ImagineBatchSize = 0
    conf.JointTrainAgent.ImagineContextLength = 0
    conf.JointTrainAgent.ImagineBatchLength = 0
    conf.JointTrainAgent.TrainDynamicsEverySteps = 0
    conf.JointTrainAgent.TrainAgentEverySteps = 0
    conf.JointTrainAgent.SaveEverySteps = 0

    conf.defrost()
    conf.merge_from_file(args.config_path)
    # conf.freeze()
    conf.Models.WorldModel.Use_amp = args.use_amp
    conf.Models.WorldModel.Use_cg = args.use_cg

    conf.Models.WorldModel.Mamba_d_model = args.d_model
    conf.Models.WorldModel.Mamba_n_layer = args.n_layer
    conf.Models.WorldModel.Mamba_d_state = args.d_state    

    conf.Models.Agent.Model = args.policy

    return conf
