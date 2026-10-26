from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import datetime
import argparse
import yaml
import logging
import os

from model import Detector
from common import *


@rank_zero_only
def get_model_path(model, pretrained, scheduler, scheduler_params, path):
    if not os.path.exists(path):
        os.makedirs(path)
    path_model = os.path.join(path, model)
    if not os.path.exists(path_model):
        os.mkdir(path_model)
    if model != "own":
        path_pretrained = os.path.join(
            path_model, "not-" * (not pretrained) + "pretrained"
        )
        if not os.path.exists(path_pretrained):
            os.mkdir(path_pretrained)
    else:
        path_pretrained = path_model
    
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if scheduler:
        stamp += f"_{scheduler}_{'-'.join(par for par in scheduler_params)}"

    save_path = os.path.join(path_pretrained, stamp)
    if os.path.exists(save_path):
        for i in range(1, 10):
            if not os.path.exists(save_path + f"-{i}"):
                save_path += f"-{i}"
                break
    os.mkdir(save_path)
    return save_path


def get_callbacks(log_path, meta=False):
    file_name =\
        "{epoch:02d}-{v_loss:.3f}-{v_acc_d:.2f}-{v_acc_a:.2f}-{v_acc_m:.2f}"\
        if meta else "{epoch:02d}-{v_loss:.3f}-{v_acc:.3f}"
    val_callback = ModelCheckpoint(
        monitor="v_loss",
        save_top_k=2,
        dirpath=log_path,
        filename=file_name,
    )
    last_callback = ModelCheckpoint(
        monitor="epoch",
        mode="max",
        dirpath=log_path,
        filename=file_name,
    )
    lr_monitor = LearningRateMonitor()

    return [val_callback, last_callback, lr_monitor]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", '-e', type=int, default=100)
    parser.add_argument("--batch", '-b', type=int, default=256)
    parser.add_argument("--learning-rate", '-lr', type=float, default=1e-4)
    parser.add_argument("--momentum", '-m', type=float, default=1e-2)
    parser.add_argument("--cls-type", '-t', type=str, default="classes")
    parser.add_argument("--quantiles-path", '-q', type=str,
                        default="quantiles.csv")
    parser.add_argument("--classes", '-cl', type=str, default=None)
    parser.add_argument("--datasets", '-ds', nargs="+", default=["koniq"])
    parser.add_argument("--presets", '-pres', nargs="*", default=[])
    parser.add_argument("--metrics", '-metr', nargs="*", default=[])
    parser.add_argument("--blacklisted-presets", '-bp', nargs="*", default=[])
    parser.add_argument("--blacklisted-metrics", '-bm', nargs="*", default=[])
    parser.add_argument("--balanced-sampler", '-bs', action="store_true")
    parser.add_argument("--save-path", '-p', type=str, default="classes")
    parser.add_argument("--test-freq", '-tf', type=int, default=0)
    parser.add_argument("--test-dataset", '-td', type=str, default="koniq")
    parser.add_argument("--binary", '-B', action="store_true")
    parser.add_argument("--dist-ratio", '-dr', type=float, default=1.)
    parser.add_argument("--train-ratio", '-tr', type=float, default=0.85)
    parser.add_argument("--val-ratio", '-vr', type=float, default=0.15)
    parser.add_argument("--test-ratio", '-Tr', type=float, default=1.)
    parser.add_argument("--add-clean", '-C', nargs="*", default=[])
    parser.add_argument("--no-clean-from-attacked", '-nca', action="store_true")
    parser.add_argument("--resize", '-R', action="store_true")
    parser.add_argument("--resize-size", '-r', type=int, nargs=2,
                        default=[384, 512])
    parser.add_argument("--add-clean-smart-resize", '-CR', action="store_true")
    parser.add_argument("--augment-resize", '-ar', action="store_true")
    parser.add_argument("--augment-crop", '-ac', action="store_true")
    parser.add_argument("--augment-clean", '-acl', action="store_true")
    parser.add_argument("--sharpen", '-S', action="store_true")
    parser.add_argument("--scheduler", '-s', type=str, default=None)
    parser.add_argument("--scheduler-params", '-sp', nargs="*", default=[])
    parser.add_argument("--model", '-M', type=str, default="own")
    parser.add_argument("--criterion", '-c', type=str, default="CEL")
    parser.add_argument("--pretrained", '-pret', action="store_true")
    parser.add_argument("--devices", '-d', type=int, default=1)
    parser.add_argument("--num-workers", '-w', type=int, default=16)
    parser.add_argument("--limit-train-batches", '-tb', type=float, default=1.)
    parser.add_argument("--limit-val-batches", '-vb', type=float, default=1.)
    parser.add_argument("--limit-test-batches", '-Tb', type=float, default=1.)
    parser.add_argument("--debug", '-D', action="store_true")
    args = parser.parse_args()

    from dataset.framework import ImageDataset
    
    model_path = MODEL_PATH_FRAMEWORK
    if args.cls_type == "classes":
        classes = CLASSES_FRAMEWORK
    elif args.cls_type == "metrics":
        classes = METRICS
        model_path = os.path.join(model_path, "metrics")
    elif args.cls_type == "meta":
        classes = {
            "attacks": CLASSES_FRAMEWORK,
            "metrics": METRICS
        }
        model_path = os.path.join(model_path, "meta")
    else:
        raise RuntimeError(f"No classification type {args.cls_type}")
    
    if args.classes:
        if args.cls_type == "meta":
            raise RuntimeError(
                "Custom classes not implemented for meta-classification"
            )
        to_remove = []
        for cl, label in zip(classes.keys(), args.classes):
            if label.isalnum():
                classes[cl] = int(label, base=36)
            else:
                to_remove.append(cl)
        for cl in to_remove:
            del classes[cl]
    
    if len(args.datasets) > 1:
        model_path = os.path.join(
            model_path, "cv",
            "-".join([dataset.split("-")[0]
                      for dataset in sorted(args.datasets)])
        )
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        for suffix in args.datasets[0].split("-")[1:]:
            if False not in [suffix in dataset.split("-")[1:]
                             for dataset in args.datasets[1:]]:
                model_path = os.path.join(model_path, suffix)
                if not os.path.exists(model_path):
                    os.mkdir(model_path)

    if args.binary:
        n_classes = 2
        for cl in classes:
            if classes[cl]:
                classes[cl] = 1
        if args.save_path == "classes":
            model_path = os.path.join(model_path, "binary")
        else:
            model_path = os.path.join(model_path, args.save_path)
    elif args.cls_type == "meta":
        n_classes = [
            2,
            len(set(classes["attacks"].values())),
            len(set(classes["metrics"].values()))
        ]
        if args.save_path != "classes":
            model_path = os.path.join(model_path, args.save_path)
    else:
        n_classes = len(set(classes.values()))
        model_path = os.path.join(model_path, args.save_path)
    
    if args.add_clean:
        model_path = os.path.join(
            model_path, "-".join([dataset.split("-")[0] for dataset
                                  in sorted(args.add_clean)])
        )
    
    resize = args.resize_size if args.resize else None

    model = Detector(
        batch=args.batch,
        lr=args.learning_rate,
        momentum=args.momentum,
        n_classes=n_classes,
        resize=resize,
        add_clean_smart_resize=args.add_clean_smart_resize,
        aug_resize=args.augment_resize,
        aug_crop=args.augment_crop,
        aug_clean=args.augment_clean,
        sharpen=args.sharpen,
        scheduler=args.scheduler,
        scheduler_params=args.scheduler_params,
        model=args.model,
        pretrained=args.pretrained,
        meta=args.cls_type == "meta",
        criterion=args.criterion,
        datasets=args.datasets,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        dist_ratio=args.dist_ratio,
        no_clean_from_attacked=args.no_clean_from_attacked,
        add_clean=args.add_clean,
        presets=args.presets,
        metrics=args.metrics,
        bl_presets=args.blacklisted_presets,
        bl_metrics=args.blacklisted_metrics,
        balanced_sampler=args.balanced_sampler,
    )
    
    log_path = get_model_path(args.model, args.pretrained, args.scheduler,
                              args.scheduler_params, model_path)
    if args.criterion == "TML":
        log_path_temp = log_path
        log_path = os.path.join(log_path, "pretrain")

    def rank_zero_filter(record):
        blacklist = [
            "PU available: ",
            "(limit_train_batches=1.0)",
            "(limit_val_batches=1.0)",
            "`Trainer.fit` stopped: `max_epochs=",
        ]
        return [filter for filter in blacklist if filter in record.getMessage()
                ] == []
    logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(
        rank_zero_filter
    )
    
    logger = TensorBoardLogger(log_path)
    callbacks = get_callbacks(log_path, meta=args.cls_type == "meta")

    if args.test_freq == 0:
        trainer = Trainer(
            logger=logger,
            max_epochs=args.epochs,
            callbacks=callbacks,
            accelerator="gpu",
            devices=args.devices,
            limit_val_batches=args.limit_val_batches,
            limit_train_batches=args.limit_train_batches,
            fast_dev_run=args.debug,
            enable_progress_bar=False,
        )
        trainer.fit(
            model,
            model.train_dataloader(
                ImageDataset, DATASET_PATH, classes, args.cls_type,
                args.quantiles_path, args.num_workers
            ),
            model.validation_dataloader(
                ImageDataset, DATASET_PATH, classes, args.cls_type,
                args.quantiles_path, args.num_workers
            )
        )
    else:
        trainer = Trainer(
            logger=logger,
            max_epochs=min(args.test_freq, args.epochs),
            callbacks=callbacks,
            accelerator="gpu",
            devices=args.devices,
            limit_val_batches=args.limit_val_batches,
            limit_train_batches=args.limit_train_batches,
            limit_test_batches=args.limit_test_batches,
            fast_dev_run=args.debug,
            enable_progress_bar=False,
        )
        trainer.fit(
            model,
            model.train_dataloader(
                ImageDataset, DATASET_PATH, classes, args.cls_type,
                args.quantiles_path, args.num_workers
            ),
            model.validation_dataloader(
                ImageDataset, DATASET_PATH, classes, args.cls_type,
                args.quantiles_path, args.num_workers
            )
        )
        for epochs in range(min(2 * args.test_freq, args.epochs),
                            args.epochs + 1, args.test_freq):
            trainer.test(
                dataloaders=model.test_dataloader(
                    ImageDataset, DATASET_PATH, classes, args.cls_type,
                    args.quantiles_path, args.num_workers
                ),
                ckpt_path=callbacks[1].best_model_path
            )
            trainer = Trainer(
                logger=logger,
                max_epochs=epochs,
                callbacks=callbacks,
                accelerator="gpu",
                devices=args.devices,
                limit_val_batches=args.limit_val_batches,
                limit_train_batches=args.limit_train_batches,
                limit_test_batches=args.limit_test_batches,
                fast_dev_run=args.debug,
                enable_progress_bar=False,
            )
            trainer.fit(
                model,
                model.train_dataloader(
                    ImageDataset, DATASET_PATH, classes, args.cls_type,
                    args.quantiles_path, args.num_workers
                ),
                model.validation_dataloader(
                    ImageDataset, DATASET_PATH, classes, args.cls_type,
                    args.quantiles_path, args.num_workers
                ),
                ckpt_path=callbacks[1].best_model_path,
            )


    if args.criterion == "TML":
        best_model_path = [
            callback for callback in trainer.callbacks
            if hasattr(callback, "monitor") and callback.monitor == "v_loss"
        ][0].best_model_path

        params_path = os.path.join(
            log_path, "lightning_logs", "version_0", "hparams.yaml"
        )
        with open(params_path, "r") as params_file:
            params = yaml.safe_load(params_file)
        
        params["batch"] *= 2
        params["lr"] *= 0.2
        params["criterion"] = "CEL"

        model_cls = Detector.load_from_checkpoint(
            best_model_path, **params
        )

        log_path = os.path.join(log_path_temp, "train")

        if n_classes == 2:
            epochs = args.epochs // 10
        else:
            epochs = args.epochs // 4

        trainer = Trainer(
            logger=TensorBoardLogger(log_path),
            max_epochs=epochs,
            callbacks=get_callbacks(log_path, meta=args.cls_type == "meta"),
            accelerator="gpu",
            devices=args.devices,
            limit_val_batches=args.limit_val_batches,
            limit_train_batches=args.limit_train_batches,
            fast_dev_run=args.debug,
            enable_progress_bar=False,
        )
        trainer.fit(
            model_cls,
            model_cls.train_dataloader(
                ImageDataset, DATASET_PATH, classes, args.cls_type,
                args.quantiles_path, args.num_workers
            ),
            model_cls.validation_dataloader(
                ImageDataset, DATASET_PATH, classes, args.cls_type,
                args.quantiles_path, args.num_workers
            )
        )
