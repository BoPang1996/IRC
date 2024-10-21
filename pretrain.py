from mox_util.setup import setup

setup()

import os
os.environ['OMP_NUM_THREADS'] = str(4)
import argparse
import torch.nn as nn
import MinkowskiEngine as ME
import pytorch_lightning as pl
from utils.read_config import generate_config
from pretrain.model_builder import make_model
from pytorch_lightning.plugins import DDPPlugin
from pretrain.lightning_trainer import LightningPretrain
from pretrain.lightning_datamodule import PretrainDataModule
from pretrain.lightning_trainer_spconv import LightningPretrainSpconv
from pytorch_lightning import loggers as pl_loggers


def main():
    """
    Code for launching the pretraining
    """
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default="config/slidr_minkunet.yaml", help="specify the config for training"
    )
    parser.add_argument(
        "--resume_path", type=str, default=None, help="provide a path to resume an incomplete training"
    )

    parser.add_argument(
        "--rank",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--world_size",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )

    args, unknown_args = parser.parse_known_args()
    config = generate_config(args.cfg_file)
    if args.resume_path:
        config['resume_path'] = args.resume_path
    for u_arg in unknown_args:
        if '=' in u_arg:
            u_arg = u_arg.strip().split('=')
            if u_arg[0][:2] == '--':
                u_arg[0] = u_arg[0][2:]
            elif u_arg[0][0] == '-':
                u_arg[0] = u_arg[0][1:]
            config[u_arg[0]] = u_arg[1]
        else:
            config[u_arg] = True

    if os.environ.get("LOCAL_RANK", 0) == 0:
        print(
            "\n" + "\n".join(list(map(lambda x: f"{x[0]:20}: {x[1]}", config.items())))
        )

    dm = PretrainDataModule(config)
    model_points, model_images = make_model(config)
    if config["num_gpus"] > 1:
        model_points = [ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model) for model in model_points]
        model_images = nn.SyncBatchNorm.convert_sync_batchnorm(model_images)
    if config["model_points"] == "minkunet":
        module = LightningPretrain(model_points, model_images, config)
    elif config["model_points"] == "voxelnet":
        module = LightningPretrainSpconv(model_points[0], model_images, config)
    path = config["working_dir"]
    address = args.init_method.split(":")
    # os.environ["MASTER_PORT"] = address[-1]
    # os.environ["MASTER_ADDR"] = ':'.join(address[:-1])
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["NODE_RANK"] = str(args.rank)
    trainer = pl.Trainer(
        gpus=config["num_gpus"],
        num_nodes=int(os.environ["WORLD_SIZE"]),
        accelerator="gpu",
        default_root_dir=path if 's3://' not in path else os.path.join('/cache/', os.path.basename(path)),
        max_epochs=config["num_epochs"],
        strategy='ddp',
        # plugins=DDPPlugin(find_unused_parameters=False),
        num_sanity_val_steps=0,
        resume_from_checkpoint=config["resume_path"],
        check_val_every_n_epoch=1,
        logger=[pl_loggers.CSVLogger(save_dir=path if 's3://' not in path else os.path.join('/cache/', os.path.basename(path)))],
    )
    print("Starting the training")
    trainer.fit(module, dm)
    if 's3://' in path:
        import moxing as mox
        print('copying logs and models')
        mox.file.copy_parallel(os.path.join('/cache/', os.path.basename(path)), path, is_processing=False)
        print('finish copying')


if __name__ == "__main__":
    main()
