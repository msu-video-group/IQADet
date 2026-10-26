from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

import datetime
import argparse
import os
import json
import yaml

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patheffects
import seaborn

from model import Detector
from common import *
from dataset.framework import Compose


def test_model(classes, cls_type, quantiles_path, transform, datasets,
               add_clean, add_clean_smart_resize, metrics, presets, model_path,
               model_name, limit_test_batches, best_model_path, params,
               test_ratio, dist_ratio, binary, classes_str, cms_path, cm_title,
               metric_names):
    test_set = ImageDataset(
        mode="test", dir=DATASET_PATH, classes=classes, cls_type=cls_type,
        quantiles_path=quantiles_path, datasets=datasets, test_ratio=test_ratio,
        dist_ratio=dist_ratio, transform=transform, add_clean=add_clean,
        add_clean_smart_resize=add_clean_smart_resize, metrics=metrics,
        presets=presets
    )

    trainer = Trainer(
        logger=TensorBoardLogger(model_path),
        accelerator="gpu",
        devices=1,
        limit_test_batches=limit_test_batches,
        enable_progress_bar=False,
    )
    model = Detector.load_from_checkpoint(
        best_model_path,
        **{**params, "pretrained": False},
    )
    trainer.test(
        model=model,
        dataloaders=DataLoader(
            test_set, num_workers=16, batch_size=model.batch
        ),
        ckpt_path=best_model_path,
    )

    if cls_type == "meta":
        results = {}
        cls_types = ("Detection", "Attacks", "Metrics")

        for cls_type_name in cls_types:
            acc = model.acc[cls_type_name].compute().detach().cpu().numpy()\
                                          .astype(float)
            roc_auc = model.roc_auc[cls_type_name].compute().detach().cpu()\
                                                  .numpy().astype(float)
            conf_matr = model.conf_matr[cls_type_name].compute().detach().cpu()\
                                                      .numpy().astype(int)
            f1_macro = model.f1_macro[cls_type_name].compute().detach().cpu()\
                                                    .numpy().astype(float)
            prec_macro = model.prec_macro[cls_type_name]\
                         .compute().detach().cpu().numpy().astype(float)
            rec_macro = model.rec_macro[cls_type_name].compute().detach().cpu()\
                                                      .numpy().astype(float)
            
            results[cls_type_name] = {
                "accuracy": acc,
                "roc_auc": roc_auc,
                "confusion_matrix": conf_matr,
                "f1_macro": f1_macro,
                "precision_macro": prec_macro,
                "recall_macro": rec_macro
            }
            
            df_cm = pd.DataFrame(conf_matr)
            
            if cls_type_name == "Detection":
                keys = ["clean", "dist"]
                plt.figure(figsize=(4, 3))
                linewidths = 0.9
                dpi = 300
            elif cls_type_name == "Attacks":
                keys = [key.split("-")[0][:8] for key
                        in CLASSES_FRAMEWORK.keys()]
                plt.figure(figsize=(11, 7))
                linewidths = 0.564
                dpi = 150
            elif cls_type_name == "Metrics":
                keys = [key.replace("-plus", "+") for key in METRICS.keys()]
                plt.figure(figsize=(11, 7))
                linewidths = 0.564
                dpi = 150

            ax = seaborn.heatmap(
                df_cm, annot=True, cbar=False, cmap="Spectral", fmt="d",
                linewidths=linewidths, linecolor="black", clip_on=False,
                annot_kws={
                    "fontsize": 13,
                    "path_effects": [
                        matplotlib.patheffects.withStroke(linewidth=2,
                                                          foreground="black")
                    ]
                },
                xticklabels=keys, yticklabels=keys,
                norm=matplotlib.colors.Normalize(vmin=0, vmax=1),
                mask=np.zeros_like(df_cm)
            )
            ax.collections[0].set_array(df_cm.div(df_cm.sum(axis=1), axis=0))
            ax.set(xlabel="predicted", ylabel="ground truth")
            ax.set_title(
                cls_type_name.title() + " - " +
                " ".join(model.model_name.split("_")).title()
                + model.pretrained * " pretrained"
            )
            
            fig_ = ax.get_figure()
            fig_.tight_layout()
            plt.close(fig_)

            cm_path = os.path.join(
                cms_path, cls_type_name.lower() + "-" + cm_title + ".png"
            )
            fig_.savefig(cm_path, dpi=dpi)
            model.loggers[0].experiment.add_figure(
                cls_type_name.title() + " - Confusion matrix",
                fig_, model.current_epoch
            )

            model_metrics = [acc, f1_macro, roc_auc, prec_macro, rec_macro]

            plt.figure(figsize=(7, 5))
            plt.bar(np.arange(len(model_metrics)), model_metrics)
            plt.ylim(0., 1.)
            plt.xticks(np.arange(len(metric_names)), metric_names, rotation=10)
            plt.yticks(np.linspace(0., 1., 11))
            
            title = (
                cls_type_name + " - " +
                " ".join(model_name.split("-")[0].split("_"))
            ).title()
            plt.title(title + (len(model_name.split("-")) > 1) * " pretrained")
            
            plt.ylabel("value")
            plt.grid()
            plt.savefig(os.path.join(
                model_path, f"{cls_type_name.lower()}-metrics.png"), dpi=300
            )
            plt.close()
        
        return results
    else:
        acc = model.acc.compute().detach().cpu().numpy().astype(float)
        roc_auc = model.roc_auc.compute().detach().cpu().numpy().astype(float)
        conf_matr =\
            model.conf_matr.compute().detach().cpu().numpy().astype(int)
        if binary:
            f1 = model.f1.compute().detach().cpu().numpy().astype(float)
            prec = model.prec.compute().detach().cpu().numpy().astype(float)
            rec = model.rec.compute().detach().cpu().numpy().astype(float)
        else:
            f1_macro =\
                model.f1_macro.compute().detach().cpu().numpy().astype(float)
            prec_macro =\
                model.prec_macro.compute().detach().cpu().numpy().astype(float)
            rec_macro =\
                model.rec_macro.compute().detach().cpu().numpy().astype(float)

        df_cm = pd.DataFrame(conf_matr)

        if binary:
            keys = ["clean", "dist"]
            plt.figure(figsize=(4, 3))
            linewidths = 0.9
            dpi = 300
        else:
            if classes_str is not None and classes_str == "0*123332222313":
                keys = ["clean", "uap", "bbox", "gradient"]
                plt.figure(figsize=(4.5, 3.5))
                linewidths = 0.6
                dpi = 200
            else:
                if cls_type == "classes":
                    keys = [key.split("-")[0][:8] for key
                            in CLASSES_FRAMEWORK.keys()]
                    if len(classes) <= 9:
                        plt.figure(figsize=(11, 7))
                        linewidths = 0.564
                        dpi = 150
                    else:
                        plt.figure(figsize=(14, 8))
                        linewidths = 0.53
                        dpi = 135
                elif cls_type == "metrics":
                    keys = [key.replace("-plus", "+") for key in METRICS.keys()]
                    plt.figure(figsize=(11, 7))
                    linewidths = 0.564
                    dpi = 150

        ax = seaborn.heatmap(
            df_cm, annot=True, cbar=False, cmap="Spectral", fmt="d",
            linewidths=linewidths, linecolor="black", clip_on=False,
            annot_kws={
                "fontsize": 13,
                "path_effects": [
                    matplotlib.patheffects.withStroke(linewidth=2,
                                                      foreground="black")
                ]
            },
            xticklabels=keys, yticklabels=keys,
            norm=matplotlib.colors.Normalize(vmin=0, vmax=1),
            mask=np.zeros_like(df_cm)
        )
        ax.collections[0].set_array(df_cm.div(df_cm.sum(axis=1), axis=0))
        ax.set(xlabel="predicted", ylabel="ground truth")
        ax.set_title(
            " ".join(model.model_name.split("_")).title() +
            model.pretrained * " pretrained"
        )
        
        fig_ = ax.get_figure()
        fig_.tight_layout()
        plt.close(fig_)

        cm_path = os.path.join(cms_path, cm_title + ".png")
        fig_.savefig(cm_path, dpi=dpi)
        model.loggers[0].experiment.add_figure(
            "Confusion matrix", fig_, model.current_epoch
        )
        
        if binary:
            model_metrics = [acc, f1, roc_auc, prec, rec]
        else:
            model_metrics = [acc, f1_macro, roc_auc, prec_macro, rec_macro]

        plt.figure(figsize=(7, 5))
        plt.bar(np.arange(len(model_metrics)), model_metrics)
        plt.ylim(0., 1.)
        plt.xticks(np.arange(len(metric_names)), metric_names, rotation=10)
        plt.yticks(np.linspace(0., 1., 11))
        
        title = " ".join(model_name.split("-")[0].split("_")).title()
        plt.title(title + (len(model_name.split("-")) > 1) * " pretrained")
        
        plt.ylabel("value")
        plt.grid()
        plt.savefig(os.path.join(model_path, "metrics.png"), dpi=300)
        plt.close()

        return model_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-file", '-p', type=str,
                        default="test/paths/models-all.json")
    parser.add_argument("--cls-type", '-t', type=str, default="classes")
    parser.add_argument("--quantiles-path", '-q', type=str,
                        default="quantiles.csv")
    parser.add_argument("--classes", '-cl', type=str, default=None)
    parser.add_argument("--datasets", '-ds', nargs="+", default=["koniq"])
    parser.add_argument("--binary", '-B', action="store_true")
    parser.add_argument("--add-clean", '-C', nargs="*", default=[])
    parser.add_argument("--test-per-metric", '-tpm', action="store_true")
    parser.add_argument("--test-per-preset", '-tpp', action="store_true")
    parser.add_argument("--presets", '-pres', nargs="*", default=[])
    parser.add_argument("--metrics", '-metr', nargs="*", default=[])
    parser.add_argument("--resize", '-r', action="store_true")
    parser.add_argument("--add-clean-smart-resize", '-CR', action="store_true")
    parser.add_argument("--dist-ratio", '-dr', type=float, default=1.)
    parser.add_argument("--test-ratio", '-tr', type=float, default=1.)
    parser.add_argument("--limit-test-batches", '-tb', type=float, default=1.)
    parser.add_argument("--no-xtick-rot", '-nxtr', action="store_true")
    parser.add_argument("--bar-width", '-w', type=int, default=0.3)
    parser.add_argument("--plot-pairs", '-P', action="store_true")
    args = parser.parse_args()
    
    if os.path.exists("/root/users/27a_leb"):
        SAVE_PATH = "/root/users/27a_leb/attack-detection/test/res"
    else:
        SAVE_PATH = "test/res"

    from dataset.framework import ImageDataset
    if args.cls_type == "classes":
        classes = CLASSES_FRAMEWORK
    elif args.cls_type == "metrics":
        classes = METRICS
    elif args.cls_type == "meta":
        classes = {
            "attacks": CLASSES_FRAMEWORK,
            "metrics": METRICS
        }
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

    if args.binary:
        for cl in classes:
            if classes[cl]:
                classes[cl] = 1

    models = dict()
    with open(args.paths_file) as json_file:
        models_json = json.load(json_file)
        for model_name in models_json:
            models[model_name] = models_json[model_name]

    stamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    model_group = "_".join(
        args.paths_file.split("/")[-1].split(".")[0].split("-")[1:]
    )
    save_path = os.path.join(SAVE_PATH, stamp + "-" + model_group)
    datasets_str = "-".join([dataset.replace("-", "_") for dataset
                             in sorted(args.datasets)])
    save_path += f"-{datasets_str}"
    if args.resize:
        save_path += "-resize"
    if args.add_clean:
        save_path += "-AC_" + "_".join([
            dataset[0] + dataset[-1] for dataset in args.add_clean
        ])
    if args.dist_ratio != 1.0:
        save_path += f"-dr{args.dist_ratio:.3f}"
    os.mkdir(save_path)

    cms_path = os.path.join(save_path, "cms")
    os.mkdir(cms_path)

    metrics = []
    if args.test_per_metric:
        attacked_metric_metrics = []
    if args.test_per_preset:
        preset_metrics = []
    if args.binary:
        metric_names = ["Accuracy", "F1", "ROC AUC", "Precision", "Recall"]
    else:
        metric_names = ["Accuracy", "F1 Macro", "ROC AUC", "Precision Macro",
                        "Recall Macro"]
    if args.cls_type == "meta":
        if args.test_per_metric or args.test_per_preset:
            raise RuntimeError(
                "Testing per metric/preset is not implemented for "
                "meta-classificaton"
            )

    for model_name in models:
        model_path = models[model_name]
        best_loss = float("inf")
        for name in os.listdir(model_path):
            if name[-5:] == ".ckpt":
                loss = float(name.split("-")[1].split("=")[1])
                if loss <= best_loss:
                    best_loss = loss
                    best_model_path = os.path.join(model_path, name)
        
        params_path = os.path.join(
            model_path, "lightning_logs", "version_0", "hparams.yaml"
        )
        with open(params_path, "r") as params_file:
            params = yaml.safe_load(params_file)
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
        if args.resize:
            transform = [transforms.Resize((384, 512))] + transform
        transform = Compose(transform)

        model_path = os.path.join(save_path, model_name)
        os.mkdir(model_path)

        metrics.append(test_model(
            classes=classes, cls_type=args.cls_type,
            quantiles_path=args.quantiles_path, transform=transform,
            datasets=args.datasets, add_clean=args.add_clean,
            add_clean_smart_resize=args.add_clean_smart_resize,
            metrics=args.metrics, presets=args.presets, model_path=model_path,
            model_name=model_name, limit_test_batches=args.limit_test_batches,
            best_model_path=best_model_path, params=params,
            test_ratio=args.test_ratio, dist_ratio=args.dist_ratio,
            binary=args.binary, classes_str=args.classes, cms_path=cms_path,
            cm_title=model_name, metric_names=metric_names
        ))

        if args.test_per_metric:
            attacked_metrics = []
            dist_dir = os.path.join(DATASET_PATH, "framework",
                                    f"attacked-{args.datasets[0]}")
            preset_names = [
                name for name in sorted(os.listdir(dist_dir))
                if name[: 6] == "preset"
            ]
            for preset in preset_names:
                preset_dir = os.path.join(dist_dir, preset)
                for cl in classes:
                    if classes[cl]:
                        attack_dir = os.path.join(preset_dir, cl)

                        attacked_metrics.extend([
                            name for name in sorted(os.listdir(attack_dir))
                            if name != "niqe" and name[0] != "."
                        ])
            attacked_metrics = list(set(attacked_metrics))

            if args.metrics:
                attacked_metrics = sorted(list(
                    set(attacked_metrics) & set(args.metrics)
                ))

            model_attacked_metric_metrics = []

            for attacked_metric in attacked_metrics:
                model_path_metric = os.path.join(model_path, attacked_metric)
                os.mkdir(model_path_metric)

                model_attacked_metric_metrics.append(test_model(
                    classes=classes, cls_type=args.cls_type,
                    quantiles_path=args.quantiles_path, transform=transform,
                    datasets=args.datasets, add_clean=args.add_clean,
                    add_clean_smart_resize=args.add_clean_smart_resize,
                    metrics=[attacked_metric], presets=args.presets,
                    model_path=model_path_metric, model_name=model_name,
                    limit_test_batches=args.limit_test_batches,
                    best_model_path=best_model_path, params=params, 
                    test_ratio=args.test_ratio, dist_ratio=args.dist_ratio,
                    binary=args.binary, classes_str=args.classes,
                    cms_path=model_path_metric, cm_title="cm",
                    metric_names=metric_names
                ))

            model_attacked_metric_metrics = np.array(
                model_attacked_metric_metrics
            )

            df = pd.DataFrame(model_attacked_metric_metrics,
                              index=attacked_metrics, columns=metric_names)
            df.to_csv(os.path.join(model_path, "attacked_metrics-metrics.csv"))

            attacked_metric_metrics.append(model_attacked_metric_metrics)
            
            plt.figure(figsize=(9, 8))
            df.plot()
            plt.ylim(0., 1.)

            plt.yticks(np.linspace(0., 1., 11))
            plt.xticks(ticks=np.arange(len(attacked_metrics)))
            if len(attacked_metrics) > 3:
                plt.xticks(rotation=45)

            title = " ".join(model_name.split("-")[0].split("_")).title()
            plt.title(title + (len(model_name.split("-")) > 1) * " pretrained" +
                      " attacked metrics")
            plt.ylabel("value")
            plt.xlabel("attacked metric")
            plt.grid()
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(
                os.path.join(model_path, "attacked_metrics-metrics.png"),
                dpi=200
            )
            plt.close()

        if args.test_per_preset:
            dist_dir = os.path.join(DATASET_PATH, "framework",
                                    f"attacked-{args.datasets[0]}")
            presets = [
                name for name in sorted(os.listdir(dist_dir))
                if name[: 6] == "preset"
            ]

            if args.presets:
                presets = sorted(list(set(presets) & set(args.presets)))

            model_preset_metrics = []

            for preset in presets:
                model_path_preset = os.path.join(model_path, preset)
                os.mkdir(model_path_preset)

                model_preset_metrics.append(test_model(
                    classes=classes, cls_type=args.cls_type,
                    quantiles_path=args.quantiles_path, transform=transform,
                    datasets=args.datasets, add_clean=args.add_clean,
                    add_clean_smart_resize=args.add_clean_smart_resize,
                    metrics=args.metrics, presets=[preset],
                    model_path=model_path_preset, model_name=model_name,
                    limit_test_batches=args.limit_test_batches,
                    best_model_path=best_model_path, params=params,
                    test_ratio=args.test_ratio, dist_ratio=args.dist_ratio,
                    binary=args.binary, classes_str=args.classes,
                    cms_path=model_path_preset, cm_title="cm",
                    metric_names=metric_names
                ))

            model_preset_metrics = np.array(model_preset_metrics)

            df = pd.DataFrame(model_preset_metrics, index=presets,
                              columns=metric_names)
            df.to_csv(os.path.join(model_path, "presets-metrics.csv"))

            preset_metrics.append(model_preset_metrics)
            
            plt.figure(figsize=(9, 8))
            df.plot()
            plt.ylim(0., 1.)

            plt.yticks(np.linspace(0., 1., 11))
            plt.xticks(ticks=np.arange(len(presets)))

            title = " ".join(model_name.split("-")[0].split("_")).title()
            plt.title(title + (len(model_name.split("-")) > 1) * " pretrained" +
                      " presets")
            plt.ylabel("value")
            plt.xlabel("preset")
            plt.grid()
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(model_path, "presets-metrics.png"),
                        dpi=200)
            plt.close()
    
    if args.cls_type == "meta":
        quit() # model comparison not implemented for meta-classification

    metrics = np.array(metrics)

    df = pd.DataFrame(metrics, index=models.keys(), columns=metric_names)
    df.to_csv(os.path.join(save_path, "metrics.csv"))

    if args.plot_pairs:
        plot_own = list(models.keys())[0][:3] == "own"
        x = np.arange(len(metrics) // 2) + plot_own
        labels = [key for key in models.keys() if "-" not in key]
    for i in range(len(metric_names)):
        plt.figure(figsize=(9, 8))
        if args.plot_pairs:
            if plot_own:
                plt.bar([0], [metrics[0, i]], args.bar_width,
                        label="_nolegend_")
            plt.bar(x - args.bar_width / 2, metrics[plot_own + 1:: 2, i],
                    args.bar_width, label="Pretrained")
            plt.bar(x + args.bar_width / 2, metrics[plot_own:: 2, i],
                    args.bar_width, label="Not pretrained")
        else:
            plt.bar(np.arange(len(metrics)), metrics[:, i], args.bar_width,
                    label="_nolegend_")
        
        plt.axhline(metrics[:, i].max(), color="blue", label="Best")
        if metric_names[i] == "ROC AUC":
            plt.axhline(0.5, linestyle="--", color="red", label="Random")
        
        plt.ylim(0., 1.)

        if args.plot_pairs:
            plt.xticks(np.arange(len(metrics) // 2 + plot_own), labels)
            if len(metrics) // 2 + plot_own > 2 and not args.no_xtick_rot:
                plt.xticks(rotation=70)
        else:
            plt.xticks(np.arange(len(metrics)), models.keys())
            if len(metrics) > 2 and not args.no_xtick_rot:
                plt.xticks(rotation=70)
        plt.yticks(np.linspace(0., 1., 11))

        plt.title(metric_names[i])
        plt.ylabel("value")
        plt.xlabel("model")
        plt.grid()
        plt.legend(loc="center right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{metric_names[i]}.png"), dpi=200)
        plt.close()
    
    if args.test_per_metric or args.test_per_preset:
        detailed_path = os.path.join(save_path, "detailed")
        os.mkdir(detailed_path)

    if args.test_per_metric:
        attacked_metric_metrics = np.array(attacked_metric_metrics)
        for metric_ind in range(attacked_metric_metrics.shape[-1]):
            df = pd.DataFrame(attacked_metric_metrics[..., metric_ind],
                              index=models.keys(), columns=attacked_metrics)
            df.to_csv(os.path.join(
                detailed_path,
                f"attacked_metrics-{metric_names[metric_ind]}.csv"
            ))

            plt.figure(figsize=(9, 8))
            df.T.plot()
            plt.ylim(0., 1.)

            plt.yticks(np.linspace(0., 1., 11))
            plt.xticks(ticks=np.arange(attacked_metric_metrics.shape[1]))
            if attacked_metric_metrics.shape[1] > 3:
                plt.xticks(rotation=45)

            plt.title(f"Attacked metrics {metric_names[metric_ind]}")
            plt.ylabel("value")
            plt.xlabel("attacked metric")
            plt.grid()
            if len(models) > 5:
                plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left",
                           borderaxespad=0, fontsize="small")
            else:
                plt.legend(loc="lower left", fontsize="small")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    detailed_path,
                    f"attacked_metrics-{metric_names[metric_ind]}.png"
                ),
                dpi=200
            )
            plt.close()

    if args.test_per_preset:
        preset_metrics = np.array(preset_metrics)
        for metric_ind in range(preset_metrics.shape[-1]):
            df = pd.DataFrame(preset_metrics[..., metric_ind],
                              index=models.keys(), columns=presets)
            df.to_csv(os.path.join(
                detailed_path, f"presets-{metric_names[metric_ind]}.csv"
            ))

            plt.figure(figsize=(9, 8))
            df.T.plot()
            plt.ylim(0., 1.)

            plt.yticks(np.linspace(0., 1., 11))
            plt.xticks(ticks=np.arange(preset_metrics.shape[1]))

            plt.title(f"Presets {metric_names[metric_ind]}")
            plt.ylabel("value")
            plt.xlabel("preset")
            plt.grid()
            if len(models) > 4:
                plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left",
                           borderaxespad=0, fontsize="small")
            else:
                plt.legend(loc="lower left", fontsize="small")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    detailed_path, f"presets-{metric_names[metric_ind]}.png"
                ),
                dpi=200
            )
            plt.close()
