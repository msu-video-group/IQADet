import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from PIL import Image
import pandas as pd

import os
import random


class ImageDataset(Dataset):
    def __init__(self, mode, dir, classes, cls_type="classes",
                 quantiles_path=None, datasets=["koniq"], train_ratio=0.85,
                 val_ratio=0.15, test_ratio=1., dist_ratio=1., output="sample",
                 transform=None, no_clean_from_attacked=False, add_clean=[],
                 add_clean_smart_resize=False, metrics=None, presets=None,
                 bl_presets=None, bl_metrics=None, crop=False,
                 max_quantile_ind=100):
        super().__init__()
        self.classes = classes
        self.cls_type = cls_type
        self.datasets = datasets
        self.output = output
        self.add_clean_smart_resize = add_clean_smart_resize
        self.max_quantile_ind = max_quantile_ind
        self._transform = transform

        dir = os.path.join(dir, "framework")
        dist_dirs = [os.path.join(dir, f"attacked-{dset}" + crop * "-cropped")
                     for dset in datasets]
        clean_dirs = [os.path.join(dir, f"clean-{dset}") for dset in datasets]
        clean_dirs *= not no_clean_from_attacked
        
        for clean_dataset in add_clean:
            clean_dirs.append(os.path.join(dir, f"clean-{clean_dataset}"))
        
        self._items = {
            "path": [], "dataset": [], "preset": [], "attack": [], "metric": []
        }

        for dist_dir in dist_dirs:
            preset_names = [
                name for name in sorted(os.listdir(dist_dir))
                if name[: 6] == "preset"
            ]
            if presets:
                preset_names = list(set(preset_names) & set(presets))
            if bl_presets:
                preset_names = list(set(preset_names) - set(bl_presets))
            
            for preset in preset_names:
                preset_dir = os.path.join(dist_dir, preset)
                self._get_items(preset_dir, preset,
                                dist_dir.split("-")[1], mode, classes, cls_type,
                                train_ratio, val_ratio, test_ratio, dist_ratio,
                                metrics, bl_metrics)

        for clean_dir in clean_dirs:
            clean_names = sorted(os.listdir(clean_dir))

            if mode == "train":
                img_names = clean_names[:int(train_ratio * len(clean_names))]
            elif mode == "val":
                img_names = clean_names[
                    int(train_ratio * len(clean_names)):
                    int((train_ratio + val_ratio) * len(clean_names))
                ]
            elif mode == "test":
                img_names = clean_names[
                    (not no_clean_from_attacked) * int(
                        (1 - test_ratio) * len(clean_names)
                    ):
                ]

            for img_name in img_names:
                img_path = os.path.join(clean_dir, img_name)
                self._items["path"].append(img_path)
                self._items["dataset"].append(clean_dir.split("-")[1])
                self._items["preset"].append("clean")
                self._items["attack"].append("clean")
                self._items["metric"].append("clean")
        
        self._items = pd.DataFrame(self._items)
        self.index = self._items.index.values
    
        if self.cls_type == "meta":
            self.quantiles = pd.read_csv(
                os.path.join(dir, quantiles_path), dtype={"file_name": str}
            )[
                ["file_name", "dataset", "preset", "attack", "metric"] +
                [
                    metric for metric in self.classes["metrics"].keys()
                    if metric != "clean"
                ]
            ].set_index(["file_name", "dataset", "preset", "attack", "metric"])
        
        if self.add_clean_smart_resize:
            first_per_dataset = self._items.groupby("dataset").first()
            dataset_counts = self._items["dataset"].value_counts()
            self.size_cnts = {}
            for dataset in datasets:
                example_img_path = first_per_dataset.at[dataset, "path"]
                dataset_size = Image.open(example_img_path).size[::-1]
                if dataset_size not in self.size_cnts:
                    self.size_cnts[dataset_size] = 0
                self.size_cnts[dataset_size] += dataset_counts[dataset]
            
            self.add_clean_transforms = [CropResizeCleanSmart(size) for size
                                         in self.size_cnts]
                
            counts = np.array(list(self.size_cnts.values()))
            probabilities = counts / sum(self.size_cnts.values())
            self.probabilities_cumsum = np.cumsum(probabilities)
            
            self.size_indices = []
            for idx in range(len(self._items)):
                dataset = os.path.basename(self._items.iloc[idx]["dataset"])
                if dataset in datasets:
                    self.size_indices.append(0)
                else:
                    filename = os.path.basename(self._items.iloc[idx]["path"])
                    self.size_indices.append(self.get_size_ind(filename))
    
    def _get_items(self, preset_dir, preset, dataset_name, mode, classes,
                   cls_type, train_ratio, val_ratio, test_ratio, dist_ratio,
                   metrics, bl_metrics):
        if cls_type == "classes":
            attack_names = list(classes.keys())
        elif cls_type == "metrics":
            attack_names = [
                name for name in sorted(os.listdir(preset_dir))
                if name != "stadv" and name[0] != "."
            ]
        elif cls_type == "meta":
            attack_names = list(classes["attacks"].keys())
        for attack in attack_names:
            if attack != "clean" and attack in [
                "clean", "cadv", "cnn-attack", "grad-est", "ifgsm",
                "korhonen-et-al", "madc", "onepixel", "parsimonious",
                "patch-rs", "square-attack", "ssah", "uap", "zhang-et-al-dists",
            ]:
                attack_dir = os.path.join(preset_dir, attack)

                if cls_type == "classes":
                    metric_names = [
                        name for name in sorted(os.listdir(attack_dir))
                        if name != "niqe" and name[0] != "."
                    ]
                elif cls_type == "metrics":
                    metric_names = list(classes.keys())
                elif cls_type == "meta":
                    metric_names = list(classes["metrics"].keys())
                    
                if metrics:
                    metric_names = list(set(metric_names) &\
                                        set(metrics))
                if bl_metrics:
                    metric_names = list(set(metric_names) -\
                                        set(bl_metrics))

                for metric in metric_names:
                    if metric != "clean":
                        metric_dir = os.path.join(attack_dir, metric)
                        dir_names = sorted(os.listdir(metric_dir))

                        dir_names = dir_names[
                            :int(max(2, dist_ratio * len(dir_names)))
                        ]

                        if mode == "train":
                            img_names = dir_names[
                                :int(max(1, train_ratio * len(dir_names)))
                            ]
                        elif mode == "val":
                            img_names = dir_names[
                                int(max(1, train_ratio * len(dir_names))):
                                int(
                                    max(
                                        2,
                                        (train_ratio + val_ratio) *
                                            len(dir_names)
                                    )
                                )
                            ]
                        elif mode == "test":
                            img_names = dir_names[
                                int((1 - test_ratio) * len(dir_names)):
                            ]

                        for img_name in img_names:
                            img_path = os.path.join(metric_dir, img_name)
                            self._items["path"].append(img_path)
                            self._items["dataset"].append(dataset_name)
                            self._items["preset"].append(preset)
                            self._items["attack"].append(attack)
                            self._items["metric"].append(metric)
    def __len__(self):
        return len(self._items)
    
    def __getitem__(self, idx):
        if self.output == "triplet":
            return self._get_triplet(idx)
        else:
            return self._get_sample(idx)

    def _get_sample(self, idx):
        img_path, dataset, preset, attack, metric = self._items.iloc[idx]

        image = Image.open(img_path).convert("RGB")

        if self.cls_type == "classes":
            attacked = self.classes[attack]
        elif self.cls_type == "metrics":
            attacked = attack != "clean" # TODO: check if this is a good idea
        else:
            attacked = self.classes["attacks"][attack]
        
        if self.add_clean_smart_resize and dataset not in self.datasets:
            image, attacked = self.add_clean_transforms[self.size_indices[idx]](
                image, attacked
            )
        if self._transform:
            image, attacked = self._transform(image, attacked)
        
        if self.cls_type == "classes":
            label = self.classes[attack]
        elif self.cls_type == "metrics":
            label = self.classes[metric]
        elif self.cls_type == "meta":
            file_name = img_path.split("/")[-1]
            quantile_idxs = self.quantiles.loc[
                (file_name, dataset, preset, attack, metric),
                [
                    metric for metric in self.classes["metrics"].keys()
                    if metric != "clean"
                ]
            ].values
            quantiles_normalized = torch.tensor(
                quantile_idxs / self.max_quantile_ind
            ).float()
            
            label = (
                np.int64(attack != "clean"),
                np.int64(self.classes["attacks"][attack]),
                np.int64(self.classes["metrics"][metric])
            )
            return (image, quantiles_normalized), label
        return image, np.int64(label)

    def _get_triplet(self, idx):
        anchor_path, label, _ = self._items.iloc[idx]

        anchor_image = Image.open(anchor_path).convert("RGB")

        other_idxs = self.index != idx
        same_labels = self._items["label"][other_idxs] == label

        positive_idxs = self.index[other_idxs][same_labels]
        positive_idx = random.choice(positive_idxs)
        positive_path = self._items["path"][positive_idx]

        negative_idxs = self.index[other_idxs][~same_labels]
        negative_idx = random.choice(negative_idxs)
        negative_path = self._items["path"][negative_idx]
        
        anchor_image = Image.open(anchor_path).convert("RGB")
        positive_image = Image.open(positive_path).convert("RGB")
        negative_image = Image.open(negative_path).convert("RGB")

        if self._transform:
            anchor_image, label = self._transform(anchor_image, label)
            positive_image, _ = self._transform(positive_image, label)
            negative_image, _ = self._transform(
                negative_image, self._items.iloc[negative_idx, "label"]
            )

        return anchor_image, positive_image, negative_image, label
    
    def get_size_ind(self, filename, seed=42):
        if self.add_clean_smart_resize:
            hash_val = seed
            for char in filename:
                hash_val = (hash_val * 31 + ord(char)) & 0xFFFFFFFF
            
            golden_ratio = 0.618033988749895
            rand_val = (hash_val * golden_ratio) % 1.0
            
            return np.searchsorted(self.probabilities_cumsum, rand_val)
        else:
            return 0


class DatasetBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, batch_size, shuffle_items=False,
                 shuffle_batches=False, attacked_datasets=[]):
        self.shuffle_items = shuffle_items
        self.shuffle_batches = shuffle_batches

        self.batch_idxs = []
        df = data_source._items
                
        grouped_indices = {}
        for idx in range(len(df)):
            dataset = df.iloc[idx]["dataset"]
            if dataset in attacked_datasets:
                size_ind = 0
            else:
                size_ind = data_source.size_indices[idx]
            
            group_key = (dataset, size_ind)
            if group_key not in grouped_indices:
                grouped_indices[group_key] = []
            grouped_indices[group_key].append(idx)
        
        for indices in grouped_indices.values():
            if self.shuffle_items:
                random.shuffle(indices)
            
            for batch_start_idx in range(0, len(indices), batch_size):
                self.batch_idxs.append(
                    indices[batch_start_idx: batch_start_idx + batch_size]
                )

    def __iter__(self):
        if self.shuffle_items:
            for idxs in self.batch_idxs:
                random.shuffle(idxs)
        if self.shuffle_batches:
            random.shuffle(self.batch_idxs)
        return iter(self.batch_idxs)

    def __len__(self):
        return len(self.batch_idxs)


class CropIndexSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, shuffle=False):
        self.shuffle = shuffle

        self.crop_options = []
        df = data_source._items
        prev_img_name = None
        for ind in range(len(df)):
            path_before_format = df.iloc[ind]["path"].split(".")[-2]
            if (len(path_before_format) > 7 and
                path_before_format[-7: -2] == "-crop"):
                crop_ind = int(path_before_format[-2:])
                img_name = path_before_format[: -7]
                if img_name == prev_img_name:
                    self.crop_options[-1].append(ind)
                else:
                    self.crop_options.append([ind])
                prev_img_name = img_name
            else:
                self.crop_options.append([ind])

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.crop_options)
        indeces = [random.choice(idxs) for idxs in self.crop_options]

        return iter(indeces)

    def __len__(self):
        return len(self.crop_options)


class BalancedIndexSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, shuffle=False):
        self.shuffle = shuffle

        self.indeces = []
        df = data_source._items
        if data_source.cls_type == "meta":
            df = df.assign(
                label=df["attack"].map(data_source.classes["attacks"])
            )
        else:
            column = "metric" if data_source.cls_type == "metrics" else "attack"
            df = df.assign(label=df[column].map(data_source.classes))
        labels_idxs = []
        max_label_samples = 0
        for label in df["label"].unique():
            labels_idxs.append(df.index[df["label"] == label].tolist())
            max_label_samples = max(max_label_samples, len(labels_idxs[-1]))
        for label_idxs in labels_idxs:
            label_samples = len(label_idxs)
            for label_samples_added in range(0, max_label_samples,
                                             label_samples):
                self.indeces.extend(label_idxs[
                    : min(label_samples,
                          max_label_samples - label_samples_added)
                ])

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indeces)

        return iter(self.indeces)

    def __len__(self):
        return len(self.indeces)


class BalancedDatasetBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, batch_size, shuffle_items=False,
                 shuffle_batches=False):
        self.shuffle_items = shuffle_items
        self.shuffle_batches = shuffle_batches

        indeces = []
        df = data_source._items
        if data_source.cls_type == "meta":
            df = df.assign(
                label=df["attack"].map(data_source.classes["attacks"])
            )
        else:
            column = "metric" if data_source.cls_type == "metrics" else "attack"
            df = df.assign(label=df[column].map(data_source.classes))
        labels_idxs = []
        max_label_samples = 0
        for label in df["label"].unique():
            df_label = (df["label"] == label)
            labels_idxs.append(list(zip(
                df.index[df_label].tolist(), df[df_label]["dataset"].tolist()
            )))
            max_label_samples = max(max_label_samples, len(labels_idxs[-1]))
        for label_idxs in labels_idxs:
            label_samples = len(label_idxs)
            for label_samples_added in range(0, max_label_samples,
                                             label_samples):
                indeces.extend(label_idxs[
                    : min(label_samples,
                          max_label_samples - label_samples_added)
                ])

        df_balanced = pd.DataFrame(indeces, columns=["idx", "dataset"])

        self.batch_idxs = []
        for dataset in df_balanced["dataset"].unique():
            dataset_idxs = df_balanced[df_balanced["dataset"] == dataset
                                       ]["idx"].tolist()
            if shuffle_items:
                random.shuffle(dataset_idxs)
            for batch_start_idx in range(0, len(dataset_idxs), batch_size):
                self.batch_idxs.append(
                    dataset_idxs[batch_start_idx: batch_start_idx + batch_size]
                )

    def __iter__(self):
        if self.shuffle_items:
            for idxs in self.batch_idxs:
                random.shuffle(idxs)
        if self.shuffle_batches:
            random.shuffle(self.batch_idxs)
        return iter(self.batch_idxs)

    def __len__(self):
        return len(self.batch_idxs)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def forward(self, img, label=0):
        for transform in self.transforms:
            if getattr(
                transform, "__module__", None
            ) == "torchvision.transforms.transforms":
                img, label = transform(img), label
            else:
                img, label = transform(img, label)
        return img, label
        
    def __call__(self, img, label=0):
        return self.forward(img, label)
    
    def __repr__(self):
        return self.__class__.__name__ + f"({self.transforms})"


class GaussianNoise(nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean
    
    def forward(self, img):
        return img + torch.randn(img.size()) * self.std + self.mean
        
    def __call__(self, img):
        return self.forward(img)
    
    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"


class Sharpen(nn.Module):
    def __init__(self, sharpness_factor=9.):
        super().__init__()
        self.sharpness_factor = sharpness_factor
    
    def forward(self, img):
        return transforms.functional.adjust_sharpness(img,
                                                      self.sharpness_factor)
        
    def __call__(self, img):
        return self.forward(img)
    
    def __repr__(self):
        return str(
            self.__class__.__name__ + f"(sharpness_factor=" +
            f"{self.sharpness_factor})"
        )


class AugmentClean(object):
    def __init__(self):
        self.transform = transforms.RandomApply([
            transforms.RandomOrder([
                transforms.RandomApply([transforms.RandomRotation(180)]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]),
            transforms.RandomApply([transforms.GaussianBlur(5)]),
            transforms.RandomApply([GaussianNoise()]),
        ], p=0.4)

    def forward(self, img, label=1):
        if label:
            return img, label
        return self.transform(img), 0
        
    def __call__(self, img, label=1):
        return self.forward(img, label)
    
    def __repr__(self):
        return self.__class__.__name__ + "()"


class CropClean(object):
    def __init__(self, size):
        self.size = size
        self.transform = CropReduce(size)

    def forward(self, img, label=1):
        if label:
            return img, label
        return self.transform(img), 0
        
    def __call__(self, img, label=1):
        return self.forward(img, label)
    
    def __repr__(self):
        return self.__class__.__name__ + f"(size={self.size})"


class CropReduce(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.transform = transforms.RandomCrop(size)
    
    def forward(self, img):
        if min(img.shape) < self.size:
            return img
        return self.transform(img)
        
    def __call__(self, img):
        return self.forward(img)
    
    def __repr__(self):
        return self.__class__.__name__ + f"(size={self.size})"


class CropRatio(nn.Module):
    def __init__(self, ratio=0.3):
        super().__init__()
        self.ratio = ratio
    
    def forward(self, img):
        transform = transforms.RandomCrop((int(self.ratio * img.size(-2)),
                                           int(self.ratio * img.size(-1))))
        return transform(img)
        
    def __call__(self, img):
        return self.forward(img)
    
    def __repr__(self):
        return self.__class__.__name__ + f"(ratio={self.ratio})"


class CropResizeCleanSmart(object):
    def __init__(self, size):
        self.size = size
        self.ratio = size[0] / size[1]
        self.resize = transforms.Resize(size)

    def forward(self, img, label=1):
        if label:
            return img, label
        
        if isinstance(img, Image.Image):
            width, height = img.size
        else:
            height, width = img.size(-2), img.size(-1)
        new_height = min([height, int(self.ratio * width)])
        new_width = int(new_height / self.ratio)
        crop = transforms.RandomCrop((new_height, new_width))(img)
        
        return self.resize(crop), 0
        
    def __call__(self, img, label=1):
        return self.forward(img, label)
    
    def __repr__(self):
        return self.__class__.__name__ + f"(size={self.size})"
