from mox_util.setup import setup

setup()

import os
import gc
import argparse
import MinkowskiEngine as ME
import pytorch_lightning as pl
from downstream.evaluate import evaluate
from utils.read_config import generate_config
from downstream.model_builder import make_model
from pytorch_lightning.plugins import DDPPlugin
from downstream.lightning_trainer import LightningDownstream
from downstream.lightning_datamodule import DownstreamDataModule
from downstream.dataloader_kitti import make_data_loader as make_data_loader_kitti
from downstream.dataloader_nuscenes import make_data_loader as make_data_loader_nuscenes
from pytorch_lightning import loggers as pl_loggers


def main():
    """
    Code for launching the downstream training
    """
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default="config/semseg_nuscenes.yaml", help="specify the config for training"
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
    dm = DownstreamDataModule(config)
    model = make_model(config, config["pretraining_path"])
    if config["num_gpus"] > 1:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    module = LightningDownstream(model, config)
    path = config["working_dir"]
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["NODE_RANK"] = str(args.rank)
    trainer = pl.Trainer(
        gpus=config["num_gpus"],
        num_nodes=int(os.environ["WORLD_SIZE"]),
        accelerator="gpu",
        default_root_dir=path if 's3://' not in path else os.path.join('/cache/', os.path.basename(path)),
        enable_checkpointing=True,
        max_epochs=config["num_epochs"],
        strategy='ddp',
        # plugins=DDPPlugin(find_unused_parameters=False),
        num_sanity_val_steps=0,
        resume_from_checkpoint=config["resume_path"],
        check_val_every_n_epoch=1,
        logger=[pl_loggers.CSVLogger(
            save_dir=path if 's3://' not in path else os.path.join('/cache/', os.path.basename(path)))],
    )
    print("Starting the training")
    trainer.fit(module, dm)

    print("Training finished, now evaluating the results")
    del trainer
    del dm
    del module
    gc.collect()
    if config["dataset"].lower() == "nuscenes":
        phase = "verifying" if config['training'] in ("parametrize", "parametrizing") else "val"
        val_dataloader = make_data_loader_nuscenes(
            config, phase, num_threads=config["num_threads"]
        )
    elif config["dataset"].lower() == "kitti":
        val_dataloader = make_data_loader_kitti(
            config, "val", num_threads=config["num_threads"]
        )
    evaluate(model.to(0), val_dataloader, config)

    if 's3://' in path:
        import moxing as mox
        print('copying logs and models')
        mox.file.copy_parallel(os.path.join('/cache/', os.path.basename(path)), path, is_processing=False)
        print('finish copying')


if __name__ == "__main__":
    main()
