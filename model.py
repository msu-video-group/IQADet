from typing import Any
import torch.nn as nn
import torch.optim
from torchvision import models, transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CyclicLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import torchmetrics
import torchmetrics.classification
from torch.utils.data import DataLoader
from dataset.framework import *


class ConvModel(nn.Module):
    def __init__(self, channels_in=3, n_classes=1):
        super().__init__()
        self.n_classes = n_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels_in, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, n_classes)
        )
        self.sm = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)

        x = x.squeeze()

        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, self.n_classes)
        return self.sm(x)
    

class MetaClassifier(nn.Module):
    def __init__(self, nums_classes, num_quantiles, backbone, hidden_dim=512,
                 dropout=0.1):
        super().__init__()

        self.backbone = backbone
        image_features_dim = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.heads = nn.ModuleList()
        self._head_input_dim = image_features_dim + num_quantiles
        for num_classes in nums_classes:
            self.heads.append(
                self._create_head(
                    self._head_input_dim, num_classes, hidden_dim, dropout
                )
            )

    def _create_head(self, input_dim, output_dim, hidden_dim, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, images, quantiles):
        image_features = self.backbone(images)
        combined_features = torch.cat([image_features, quantiles], dim=1)
        return tuple(head(combined_features) for head in self.heads)
    

class Detector(pl.LightningModule):
    def __init__(self, batch=16, lr=1e-4, momentum=0, channels_in=3,
                 n_classes=1, resize=None, add_clean_smart_resize=False,
                 aug_resize=False, aug_crop=False, aug_clean=False,
                 sharpen=False, scheduler=None, scheduler_params=None,
                 model="own", pretrained=False, meta=False, criterion="CEL",
                 datasets=["koniq"], train_ratio=0.85, val_ratio=0.15,
                 test_ratio=1., dist_ratio=1., no_clean_from_attacked=False,
                 add_clean=[], presets=None, metrics=None, bl_presets=None,
                 bl_metrics=None, balanced_sampler=False):
        super().__init__()

        self.batch = batch
        self.lr = lr
        self.momentum = momentum
        self.channels_in = channels_in
        self.n_classes = n_classes
        self.resize = resize
        self.add_clean_smart_resize = add_clean_smart_resize
        self.aug_resize = aug_resize
        self.aug_crop = aug_crop
        self.aug_clean = aug_clean
        self.sharpen = sharpen
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.model_name = model
        self.pretrained = pretrained
        self.meta = meta
        self.criterion_name = criterion
        self.datasets = datasets
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.dist_ratio = dist_ratio
        self.no_clean_from_attacked = no_clean_from_attacked
        self.add_clean = add_clean
        self.presets = presets
        self.metrics = metrics
        self.bl_presets = bl_presets
        self.bl_metrics = bl_metrics
        self.balanced_sampler = balanced_sampler

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        self.crop_mode = self.aug_crop and not self.aug_resize
        self.dataset_batching = self.resize is None or self.aug_resize \
            or self.add_clean_smart_resize
        
        transform = []
        if self.resize is not None:
            transform += [transforms.Resize(self.resize)]
        self.transform_batch_train = []
        
        transform_val = [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
        if self.resize:
            transform_val = [transforms.Resize(self.resize)] + transform_val
        self.transform_val = Compose(transform_val)
        self.transform_batch_val = []

        self.postprocessing = []
        if self.sharpen:
            self.postprocessing += [Sharpen()]
        if self.postprocessing:
            self.postprocessing = transforms.Compose(self.postprocessing)
        
        if self.aug_resize or self.aug_crop:
            transform_batch_train = []
            if self.aug_resize:
                if self.add_clean:
                    raise NotImplementedError(
                        "Resize augmentation with add-clean is not implemented"
                    )
                transform_batch_train.append(
                    transforms.RandomChoice([
                        transforms.Resize(size)
                        for size in range(128, 128 * 6, 128)
                    ])
                )
            if self.aug_crop:
                if self.crop_mode:
                    self.transform_val = Compose([
                        transforms.Resize((384, 512)),
                        transforms.ToTensor(),
                        CropClean(128),
                        transforms.Normalize(mean=mean, std=std)
                    ])
                else:
                    transform_batch_train.append(transforms.RandomApply([
                        transforms.RandomChoice([
                            CropReduce(crop)
                            for crop in range(128, 128 * 6, 128)
                        ])
                    ], p=0.3))
            transform += [transforms.ToTensor()]
            if self.crop_mode:
                transform += [
                    CropClean(128),
                    transforms.Normalize(mean=mean, std=std)
                ]
            else:
                self.transform_batch_train = transforms.Compose(
                    transform_batch_train + [transforms.Normalize(mean=mean,
                                                                  std=std)]
                )
            if self.aug_clean:
                transform += [AugmentClean()]
        else:
            transform += [transforms.ToTensor()]
            if self.aug_clean:
                transform += [AugmentClean()]
            transform += [transforms.Normalize(mean=mean, std=std)]
        self.transform = Compose(transform)

        self.embeddings = []
        def hook(model, input, output):
            self.embeddings.append(input[0])

        if self.meta:
            n_classes = 2 # Need any integer value larger than 1
        weights = None
        if model == "own":
            self.model = ConvModel(channels_in, n_classes)
        elif model == "densenet121":
            if pretrained:
                weights = models.DenseNet121_Weights.DEFAULT
            self.model = models.densenet121(weights=weights)
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features, n_classes
            )
            if self.criterion_name == "TML":
                self.model.classifier.register_forward_hook(hook)
                for param in self.model.classifier.parameters():
                    param.requires_grad = False
        elif model == "densenet161":
            if pretrained:
                weights = models.DenseNet161_Weights.DEFAULT
            self.model = models.densenet161(weights=weights)
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features, n_classes
            )
            if self.criterion_name == "TML":
                self.model.classifier.register_forward_hook(hook)
                for param in self.model.classifier.parameters():
                    param.requires_grad = False
        elif model == "densenet169":
            if pretrained:
                weights = models.DenseNet169_Weights.DEFAULT
            self.model = models.densenet169(weights=weights)
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features, n_classes
            )
            if self.criterion_name == "TML":
                self.model.classifier.register_forward_hook(hook)
                for param in self.model.classifier.parameters():
                    param.requires_grad = False
        elif model == "densenet201":
            if pretrained:
                weights = models.DenseNet201_Weights.DEFAULT
            self.model = models.densenet201(weights=weights)
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features, n_classes
            )
            if self.criterion_name == "TML":
                self.model.classifier.register_forward_hook(hook)
                for param in self.model.classifier.parameters():
                    param.requires_grad = False
        elif model == "efficientnet_v2_s":
            if pretrained:
                weights = models.EfficientNet_V2_S_Weights.DEFAULT
            self.model = models.efficientnet_v2_s(weights=weights)
            self.model.classifier[1] = nn.Linear(1280, n_classes)
            if self.criterion_name == "TML":
                self.model.classifier[1].register_forward_hook(hook)
                for param in self.model.classifier[1].parameters():
                    param.requires_grad = False
        elif model == "efficientnet_v2_m":
            if pretrained:
                weights = models.EfficientNet_V2_M_Weights.DEFAULT
            self.model = models.efficientnet_v2_m(weights=weights)
            self.model.classifier[1] = nn.Linear(1280, n_classes)
            if self.criterion_name == "TML":
                self.model.classifier[1].register_forward_hook(hook)
                for param in self.model.classifier[1].parameters():
                    param.requires_grad = False
        elif model == "efficientnet_v2_l":
            if pretrained:
                weights = models.EfficientNet_V2_L_Weights.DEFAULT
            self.model = models.efficientnet_v2_l(weights=weights)
            self.model.classifier[1] = nn.Linear(1280, n_classes)
            if self.criterion_name == "TML":
                self.model.classifier[1].register_forward_hook(hook)
                for param in self.model.classifier[1].parameters():
                    param.requires_grad = False
        elif model == "inception_v3":
            if pretrained:
                weights = models.Inception_V3_Weights.DEFAULT
            self.model = models.inception_v3(weights=weights)
            self.model.aux_logits = False
            self.model.fc = nn.Linear(2048, n_classes)
            if self.criterion_name == "TML":
                self.model.fc.register_forward_hook(hook)
                for param in self.model.fc.parameters():
                    param.requires_grad = False
        elif model == "mobilenet_v2":
            if pretrained:
                weights = models.MobileNet_V2_Weights.DEFAULT
            self.model = models.mobilenet_v2(weights=weights)
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.model.last_channel, n_classes)
            )
            if self.criterion_name == "TML":
                self.model.classifier.register_forward_hook(hook)
                for param in self.model.classifier[-1].parameters():
                    param.requires_grad = False
        elif model == "mobilenet_v3_large":
            if pretrained:
                weights = models.MobileNet_V3_Large_Weights.DEFAULT
            self.model = models.mobilenet_v3_large(weights=weights)
            self.model.classifier[-1] = nn.Linear(1280, n_classes)
            if self.criterion_name == "TML":
                self.model.classifier[-1].register_forward_hook(hook)
                for param in self.model.classifier[-1].parameters():
                    param.requires_grad = False
        elif model == "mobilenet_v3_small":
            if pretrained:
                weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.model = models.mobilenet_v3_small(weights=weights)
            self.model.classifier[-1] = nn.Linear(1024, n_classes)
            if self.criterion_name == "TML":
                self.model.classifier[-1].register_forward_hook(hook)
                for param in self.model.classifier[-1].parameters():
                    param.requires_grad = False
        elif model == "resnet18":
            if pretrained:
                weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
            self.model.fc = nn.Linear(512, n_classes)
            if self.criterion_name == "TML":
                self.model.fc.register_forward_hook(hook)
                for param in self.model.fc.parameters():
                    param.requires_grad = False
        elif model == "resnet34":
            if pretrained:
                weights = models.ResNet34_Weights.DEFAULT
            self.model = models.resnet34(weights=weights)
            self.model.fc = nn.Linear(512, n_classes)
            if self.criterion_name == "TML":
                self.model.fc.register_forward_hook(hook)
                for param in self.model.fc.parameters():
                    param.requires_grad = False
        elif model == "resnet50":
            if pretrained:
                weights = models.ResNet50_Weights.DEFAULT
            self.model = models.resnet50(weights=weights)
            self.model.fc = nn.Linear(2048, n_classes)
            if self.criterion_name == "TML":
                self.model.fc.register_forward_hook(hook)
                for param in self.model.fc.parameters():
                    param.requires_grad = False
        elif model == "resnet101":
            if pretrained:
                weights = models.ResNet101_Weights.DEFAULT
            self.model = models.resnet101(weights=weights)
            self.model.fc = nn.Linear(2048, n_classes)
            if self.criterion_name == "TML":
                self.model.fc.register_forward_hook(hook)
                for param in self.model.fc.parameters():
                    param.requires_grad = False
        elif model == "resnet152":
            if pretrained:
                weights = models.ResNet152_Weights.DEFAULT
            self.model = models.resnet152(weights=weights)
            self.model.fc = nn.Linear(2048, n_classes)
            if self.criterion_name == "TML":
                self.model.fc.register_forward_hook(hook)
                for param in self.model.fc.parameters():
                    param.requires_grad = False
        elif model == "squeezenet":
            if pretrained:
                weights = models.SqueezeNet1_1_Weights.DEFAULT
            self.model = models.squeezenet1_1(weights=weights)
            self.model.classifier._modules["1"] = nn.Conv2d(
                512, n_classes, kernel_size=(1, 1)
            )
            if self.criterion_name == "TML":
                self.model.classifier._modules["1"].register_forward_hook(hook)
                for param in self.model.classifier._modules["1"].parameters():
                    param.requires_grad = False
        elif model == "swin_t":
            if pretrained:
                weights = models.Swin_T_Weights.DEFAULT
            self.model = models.swin_t(weights=weights)
            self.model.head = nn.Linear(768, n_classes)
            if self.criterion_name == "TML":
                self.model.head.register_forward_hook(hook)
                for param in self.model.head.parameters():
                    param.requires_grad = False
        elif model == "swin_s":
            if pretrained:
                weights = models.Swin_S_Weights.DEFAULT
            self.model = models.swin_s(weights=weights)
            self.model.head = nn.Linear(768, n_classes)
            if self.criterion_name == "TML":
                self.model.head.register_forward_hook(hook)
                for param in self.model.head.parameters():
                    param.requires_grad = False
        elif model == "swin_b":
            if pretrained:
                weights = models.Swin_B_Weights.DEFAULT
            self.model = models.swin_b(weights=weights)
            self.model.head = nn.Linear(1024, n_classes)
            if self.criterion_name == "TML":
                self.model.head.register_forward_hook(hook)
                for param in self.model.head.parameters():
                    param.requires_grad = False
        elif model == "swin_v2_t":
            if pretrained:
                weights = models.Swin_V2_T_Weights.DEFAULT
            self.model = models.swin_v2_t(weights=weights)
            self.model.head = nn.Linear(768, n_classes)
            if self.criterion_name == "TML":
                self.model.head.register_forward_hook(hook)
                for param in self.model.head.parameters():
                    param.requires_grad = False
        elif model == "swin_v2_s":
            if pretrained:
                weights = models.Swin_V2_S_Weights.DEFAULT
            self.model = models.swin_v2_s(weights=weights)
            self.model.head = nn.Linear(768, n_classes)
            if self.criterion_name == "TML":
                self.model.head.register_forward_hook(hook)
                for param in self.model.head.parameters():
                    param.requires_grad = False
        elif model == "swin_v2_b":
            if pretrained:
                weights = models.Swin_V2_B_Weights.DEFAULT
            self.model = models.swin_v2_b(weights=weights)
            self.model.head = nn.Linear(1024, n_classes)
            if self.criterion_name == "TML":
                self.model.head.register_forward_hook(hook)
                for param in self.model.head.parameters():
                    param.requires_grad = False
        elif model == "vgg11":
            if pretrained:
                weights = models.VGG11_BN_Weights.DEFAULT
            self.model = models.vgg11_bn(weights=weights)
            self.model.classifier[-1] = nn.Linear(4096, n_classes)
            if self.criterion_name == "TML":
                self.model.classifier[-1].register_forward_hook(hook)
                for param in self.model.classifier[-1].parameters():
                    param.requires_grad = False
        elif model == "vgg13":
            if pretrained:
                weights = models.VGG13_BN_Weights.DEFAULT
            self.model = models.vgg13_bn(weights=weights)
            self.model.classifier[-1] = nn.Linear(4096, n_classes)
            if self.criterion_name == "TML":
                self.model.classifier[-1].register_forward_hook(hook)
                for param in self.model.classifier[-1].parameters():
                    param.requires_grad = False
        elif model == "vgg16":
            if pretrained:
                weights = models.VGG16_BN_Weights.DEFAULT
            self.model = models.vgg16_bn(weights=weights)
            self.model.classifier[-1] = nn.Linear(4096, n_classes)
            if self.criterion_name == "TML":
                self.model.classifier[-1].register_forward_hook(hook)
                for param in self.model.classifier[-1].parameters():
                    param.requires_grad = False
        elif model == "vgg19":
            if pretrained:
                weights = models.VGG19_BN_Weights.DEFAULT
            self.model = models.vgg19_bn(weights=weights)
            self.model.classifier[-1] = nn.Linear(4096, n_classes)
            if self.criterion_name == "TML":
                self.model.classifier[-1].register_forward_hook(hook)
                for param in self.model.classifier[-1].parameters():
                    param.requires_grad = False
        else:
            raise RuntimeError(f"No model {model}")
        
        if meta:
            n_classes = self.n_classes # For correct hyperparameter record
            if self.criterion_name == "TML":
                raise RuntimeError(
                    "TML not implemented for meta-classification"
                )
            if self.crop_mode:
                raise RuntimeError(
                    "Crop mode not implemented for meta-classification"
                )
            self.model = MetaClassifier(
                self.n_classes, self.n_classes[-1] - 1, self.model
            ).to(self.device)
        
        if self.criterion_name == "CEL":
            self.criterion = nn.CrossEntropyLoss(reduction="mean")
        elif self.criterion_name == "TML":
            self.criterion = nn.TripletMarginLoss(reduction="mean")
        else:
            raise RuntimeError(f"No criterion {criterion}")

        if self.meta:
            cls_types = ("Detection", "Attacks", "Metrics")
            self.acc = nn.ModuleDict()
            self.f1_macro = nn.ModuleDict()
            self.roc_auc = nn.ModuleDict()
            self.prec_macro = nn.ModuleDict()
            self.rec_macro = nn.ModuleDict()
            self.ap = nn.ModuleDict()
            self.conf_matr = nn.ModuleDict()
            self.roc_curve = nn.ModuleDict()
            self.pr_curve = nn.ModuleDict()
            for cls_type, num_classes in zip(cls_types, self.n_classes):
                self.acc[cls_type] = torchmetrics.classification.Accuracy(
                    task="multiclass", num_classes=num_classes
                )
                self.f1_macro[cls_type] = torchmetrics.classification.F1Score(
                    task="multiclass", num_classes=num_classes, average="macro"
                )
                self.roc_auc[cls_type] = torchmetrics.classification.AUROC(
                    task="multiclass", num_classes=num_classes
                )
                self.prec_macro[cls_type] =\
                    torchmetrics.classification.Precision(
                        task="multiclass", num_classes=num_classes,
                        average="macro"
                    )
                self.rec_macro[cls_type] = torchmetrics.classification.Recall(
                    task="multiclass", num_classes=num_classes, average="macro"
                )
                self.ap[cls_type] =\
                    torchmetrics.classification.MulticlassAveragePrecision(
                        num_classes=num_classes
                    )
                self.conf_matr[cls_type] =\
                    torchmetrics.classification.ConfusionMatrix(
                        task="multiclass", num_classes=num_classes
                    )
                self.roc_curve[cls_type] =\
                    torchmetrics.classification.MulticlassROC(
                        num_classes=num_classes
                    )
                self.pr_curve[cls_type] =\
                    torchmetrics.classification.MulticlassPrecisionRecallCurve(
                        num_classes=num_classes
                    )
        else:
            if n_classes == 2:
                self.acc = torchmetrics.classification.Accuracy(
                    task="binary", num_classes=n_classes
                )
                self.f1 = torchmetrics.classification.F1Score(
                    task="binary", num_classes=n_classes
                )
                self.roc_auc = torchmetrics.classification.AUROC(
                    task="binary", num_classes=n_classes
                )
                self.prec = torchmetrics.classification.Precision(
                    task="binary", num_classes=n_classes
                )
                self.rec = torchmetrics.classification.Recall(
                    task="binary", num_classes=n_classes
                )
                self.ap = torchmetrics.classification.BinaryAveragePrecision()
                self.conf_matr = torchmetrics.classification.ConfusionMatrix(
                    task="binary", num_classes=n_classes
                )
                self.roc_curve = torchmetrics.classification.BinaryROC()
                self.pr_curve =\
                    torchmetrics.classification.BinaryPrecisionRecallCurve()
            else:
                self.acc = torchmetrics.classification.Accuracy(
                    task="multiclass", num_classes=n_classes
                )
                self.f1_macro = torchmetrics.classification.F1Score(
                    task="multiclass", num_classes=n_classes, average="macro"
                )
                self.roc_auc = torchmetrics.classification.AUROC(
                    task="multiclass", num_classes=n_classes
                )
                self.prec_macro = torchmetrics.classification.Precision(
                    task="multiclass", num_classes=n_classes, average="macro"
                )
                self.rec_macro = torchmetrics.classification.Recall(
                    task="multiclass", num_classes=n_classes, average="macro"
                )
                self.ap = torchmetrics.classification.MulticlassAveragePrecision(
                    num_classes=n_classes
                )
                self.conf_matr = torchmetrics.classification.ConfusionMatrix(
                    task="multiclass", num_classes=n_classes
                )
                self.roc_curve = torchmetrics.classification.MulticlassROC(
                    num_classes=n_classes
                )
                self.pr_curve =\
                    torchmetrics.classification.MulticlassPrecisionRecallCurve(
                        num_classes=n_classes
                    )

        if self.meta:
            self.test_preds = [[], [], []]
            self.test_gts = [[], [], []]
            self.last_test_preds = None
            self.last_test_gts = None
        else:
            self.test_preds = []
            self.test_gts = []
            self.last_test_preds = []
            self.last_test_gts = []

        #self.example_input_array = torch.Tensor(batch, channels_in, 384, 512)
        self.save_hyperparameters()

    def configure_optimizers(self):
        parameters = self.model.parameters()
        
        optimizer = torch.optim.Adam(
            parameters, lr=self.lr
        )
        if self.scheduler:
            if self.scheduler == "SLR":
                scheduler_params = {
                    "step_size": 20,
                    "gamma": 0.1,
                    "last_epoch": -1,
                }
                scheduler_class = StepLR
            elif self.scheduler == "CALR":
                scheduler_params = {
                    "T_max": 10,
                    "eta_min": 1e-6,
                    "last_epoch": -1,
                }
                scheduler_class = CosineAnnealingLR
            elif self.scheduler == "CLR":
                scheduler_params = {
                    "base_lr": 1e-6,
                    "max_lr": 1e-2,
                    "mode": "triangular2",
                }
                scheduler_class = CyclicLR
            elif self.scheduler == "RLROP":
                scheduler_params = {
                    "factor": 0.2,
                    "patience": 3,
                    "min_lr": 1e-6,
                }
                scheduler_class = ReduceLROnPlateau
            else:
                raise RuntimeError(f"No scheduler {self.scheduler}")
            
            keys = list(scheduler_params.keys())
            for i in range(min(len(scheduler_params),
                               len(self.scheduler_params))):
                param_name = keys[i]
                param_type = type(scheduler_params[param_name])
                scheduler_params[param_name] = param_type(
                    self.scheduler_params[i]
                )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler_class(optimizer, **scheduler_params),
                    "interval": "epoch",
                    "monitor": "v_loss"
                }
            }
        return optimizer
    
    def forward(self, images, quantiles=None):
        if self.meta:
            if quantiles is None:
                raise ValueError("quantiles must be provided when meta=True")
            return self.model(images, quantiles)
        return self.model(images)

    def training_step(self, train_batch, batch_idx):
        if self.criterion_name == "CEL":
            x_batch, y_batch = train_batch
            if self.transform_batch_train:
                if self.meta:
                    x_batch[0] = self.transform_batch_train(x_batch[0])
                else:
                    x_batch = self.transform_batch_train(x_batch)
            if self.postprocessing:
                if self.meta:
                    x_batch[0] = self.postprocessing(x_batch[0])
                else:
                    x_batch = self.postprocessing(x_batch)
            if self.meta:
                y_pred = self.model(*x_batch)
                loss = 0
                for meta_logits_pred, meta_logits_gt in zip(y_pred, y_batch):
                    loss += self.criterion(meta_logits_pred, meta_logits_gt)
            else:
                y_pred = self.model(x_batch)
                loss = self.criterion(y_pred, y_batch)
        elif self.criterion_name == "TML":
            anchor_batch, positive_batch, negative_batch, y_batch = train_batch
            if self.transform_batch_train:
                anchor_batch = self.transform_batch_train(anchor_batch)
                positive_batch = self.transform_batch_train(positive_batch)
                negative_batch = self.transform_batch_train(negative_batch)
            self.embeddings = []
            self.model(anchor_batch)
            self.model(positive_batch)
            self.model(negative_batch)
            anchor_emb, positive_emb, negative_emb = self.embeddings
            loss = self.criterion(anchor_emb, positive_emb, negative_emb)

        if self.criterion_name == "CEL":
            if self.meta:
                acc_detection = self.acc["Detection"](y_pred[0], y_batch[0])
                acc_attacks = self.acc["Attacks"](y_pred[1], y_batch[1])
                acc_metrics = self.acc["Metrics"](y_pred[2], y_batch[2])
                values = {
                    "t_loss": loss,
                    "t_acc_d": acc_detection,
                    "t_acc_a": acc_attacks,
                    "t_acc_m": acc_metrics
                }
            else:
                if self.n_classes == 2:
                    acc = self.acc(y_pred[:, 1], y_batch)
                else:
                    acc = self.acc(y_pred, y_batch)
                values = {"t_loss": loss, "t_acc": acc}
        elif self.criterion_name == "TML":
            values = {"t_loss": loss}
        
        self.log_dict(values, on_step=False, on_epoch=True, sync_dist=True,
                      prog_bar=True, logger=True)
        self.log("epoch", self.current_epoch)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        if self.criterion_name == "CEL":
            x_batch, y_batch = val_batch
            if self.transform_batch_val:
                if self.meta:
                    x_batch[0] = self.transform_batch_val(x_batch[0])
                else:
                    x_batch = self.transform_batch_val(x_batch)
            if self.postprocessing:
                if self.meta:
                    x_batch[0] = self.postprocessing(x_batch[0])
                else:
                    x_batch = self.postprocessing(x_batch)
            if self.meta:
                y_pred = self.model(*x_batch)
                loss = 0
                for meta_logits_pred, meta_logits_gt in zip(y_pred, y_batch):
                    loss += self.criterion(meta_logits_pred, meta_logits_gt)
            else:
                y_pred = self.model(x_batch)
                loss = self.criterion(y_pred, y_batch)
        elif self.criterion_name == "TML":
            anchor_batch, positive_batch, negative_batch, y_batch = val_batch
            if self.transform_batch_val:
                anchor_batch = self.transform_batch_val(anchor_batch)
                positive_batch = self.transform_batch_val(positive_batch)
                negative_batch = self.transform_batch_val(negative_batch)
            self.embeddings = []
            self.model(anchor_batch)
            self.model(positive_batch)
            self.model(negative_batch)
            anchor_emb, positive_emb, negative_emb = self.embeddings
            loss = self.criterion(anchor_emb, positive_emb, negative_emb)

        if self.criterion_name == "CEL":
            if self.meta:
                acc_detection = self.acc["Detection"](y_pred[0], y_batch[0])
                acc_attacks = self.acc["Attacks"](y_pred[1], y_batch[1])
                acc_metrics = self.acc["Metrics"](y_pred[2], y_batch[2])
                values = {
                    "v_loss": loss,
                    "v_acc_d": acc_detection,
                    "v_acc_a": acc_attacks,
                    "v_acc_m": acc_metrics
                }
            else:
                if self.n_classes == 2:
                    acc = self.acc(y_pred[:, 1], y_batch)
                else:
                    acc = self.acc(y_pred, y_batch)
                values = {"v_loss": loss, "v_acc": acc}
        elif self.criterion_name == "TML":
            values = {"v_loss": loss}
        
        self.log_dict(values, on_epoch=True, sync_dist=True,
                      prog_bar=True, logger=True)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        if self.criterion_name == "CEL":
            x_batch, y_batch = test_batch
            if self.postprocessing:
                if self.meta:
                    x_batch[0] = self.postprocessing(x_batch[0])
                else:
                    x_batch = self.postprocessing(x_batch)

            if self.crop_mode:
                y_pred = torch.zeros(x_batch.size(0), self.n_classes,
                                     device=self.device)
                for y in range(0, 384, 128):
                    for x in range(0, 512, 128):
                        x_batch_cropped = x_batch[..., y: y + 128, x: x + 128]
                        if self.n_classes == 2:
                            y_pred[:, 1] = torch.maximum(
                                y_pred, self.model(x_batch_cropped)[:, 1]
                            )
                        else:
                            y_pred = torch.maximum(y_pred,
                                                   self.model(x_batch_cropped))
                if self.n_classes == 2:
                    y_pred[:, 0] = 1 - y_pred[:, 1]
            elif self.meta:
                y_pred = self.model(*x_batch)
            else:
                y_pred = self.model(x_batch)

            if self.meta:
                loss = 0
                for cls_ind, (meta_logits_pred, meta_logits_gt) in enumerate(
                    zip(y_pred, y_batch)
                ):
                    loss += self.criterion(meta_logits_pred, meta_logits_gt)
                    self.test_preds[cls_ind].append(meta_logits_pred)
                    self.test_gts[cls_ind].append(meta_logits_gt)
            else:
                loss = self.criterion(y_pred, y_batch)
                self.test_preds.append(y_pred)
                self.test_gts.append(y_batch)
        elif self.criterion_name == "TML":
            raise RuntimeError(f"No testing for criterion TML")

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, _ = batch
        if self.meta:
            return self(*inputs)
        return self(inputs)

    def on_test_epoch_end(self):
        if self.meta:
            y_pred = {
                "Detection": (
                    torch.cat(self.test_preds[0]), torch.cat(self.test_gts[0])
                ),
                "Attacks": (
                    torch.cat(self.test_preds[1]), torch.cat(self.test_gts[1])
                ),
                "Metrics": (
                    torch.cat(self.test_preds[2]), torch.cat(self.test_gts[2])
                ),
            }
            values = {}
            loss = 0
            for cls_type, (y_pred_cls, y_gt_cls) in y_pred.items():
                loss += self.criterion(y_pred_cls, y_gt_cls)
                acc = self.acc[cls_type](y_pred_cls, y_gt_cls)
                roc_auc = self.roc_auc[cls_type](y_pred_cls, y_gt_cls)
                ap = self.ap[cls_type](y_pred_cls, y_gt_cls)
                self.conf_matr[cls_type](y_pred_cls, y_gt_cls)
                self.roc_curve[cls_type](y_pred_cls, y_gt_cls)
                self.pr_curve[cls_type](y_pred_cls, y_gt_cls)
                f1_macro = self.f1_macro[cls_type](y_pred_cls, y_gt_cls)
                prec_macro = self.prec_macro[cls_type](y_pred_cls, y_gt_cls)
                rec_macro = self.rec_macro[cls_type](y_pred_cls, y_gt_cls)
                values[cls_type + " Accuracy"] = acc
                values[cls_type + " F1 Macro"] = f1_macro
                values[cls_type + " ROC AUC"] = roc_auc
                values[cls_type + " AP"] = ap
                values[cls_type + " Precision Macro"] = prec_macro
                values[cls_type + " Recall Macro"] = rec_macro
            values["Loss"] = loss
                
            self.last_test_preds = self.test_preds
            self.last_test_gts = self.test_gts
            self.test_preds = []
            self.test_gts = []
        else:
            y_pred = torch.cat(self.test_preds)
            y_gt = torch.cat(self.test_gts)
            self.last_test_preds = self.test_preds
            self.last_test_gts = self.test_gts
            self.test_preds = []
            self.test_gts = []

            loss = self.criterion(y_pred, y_gt)

            if self.n_classes == 2:
                y_pred = y_pred[:, 1]
            
            acc = self.acc(y_pred, y_gt)
            roc_auc = self.roc_auc(y_pred, y_gt)
            ap = self.ap(y_pred, y_gt)
            self.conf_matr(y_pred, y_gt)
            self.roc_curve(y_pred, y_gt)
            self.pr_curve(y_pred, y_gt)
            
            if self.n_classes == 2:
                f1 = self.f1(y_pred, y_gt)
                prec = self.prec(y_pred, y_gt)
                rec = self.rec(y_pred, y_gt)
                values = {
                    "Loss": loss, "Accuracy": acc, "F1": f1, "ROC AUC": roc_auc,
                    "AP": ap, "Precision": prec, "Recall": rec
                }
            else:
                f1_macro = self.f1_macro(y_pred, y_gt)
                prec_macro = self.prec_macro(y_pred, y_gt)
                rec_macro = self.rec_macro(y_pred, y_gt)
                values = {
                    "Loss": loss, "Accuracy": acc, "F1 Macro": f1_macro,
                    "ROC AUC": roc_auc, "AP": ap, "Precision Macro": prec_macro,
                    "Recall Macro": rec_macro
                }

        self.log_dict(values)

    def train_dataloader(self, dataset, dir, classes, cls_type="classes",
                         quantiles_path=None, num_workers=16):
        if self.criterion_name == "CEL":
            output = "sample"
        elif self.criterion_name == "TML":
            output = "triplet"
        
        train_set = dataset(
            mode="train", dir=dir, classes=classes, cls_type=cls_type,
            quantiles_path=quantiles_path, datasets=self.datasets,
            train_ratio=self.train_ratio, val_ratio=self.val_ratio,
            test_ratio=self.test_ratio, dist_ratio=self.dist_ratio,
            output=output, transform=self.transform,
            no_clean_from_attacked=self.no_clean_from_attacked,
            add_clean=self.add_clean,
            add_clean_smart_resize=self.add_clean_smart_resize,
            presets=self.presets, metrics=self.metrics, 
            bl_presets=self.bl_presets, bl_metrics=self.bl_metrics,
            crop=self.crop_mode,
        )
        if self.dataset_batching and not self.balanced_sampler:
            train_loader = DataLoader(
                train_set, num_workers=num_workers,
                batch_sampler=DatasetBatchSampler(
                    data_source=train_set, batch_size=self.batch,
                    shuffle_items=True, shuffle_batches=True,
                    attacked_datasets=self.datasets
                )
            )
        elif self.crop_mode:
            train_loader = DataLoader(
                train_set, num_workers=num_workers, batch_size=self.batch,
                sampler=CropIndexSampler(data_source=train_set, shuffle=True)
            )
        elif self.balanced_sampler:
            if self.dataset_batching:
                train_loader = DataLoader(
                    train_set, num_workers=num_workers,
                    batch_sampler=BalancedDatasetBatchSampler(
                        data_source=train_set, batch_size=self.batch,
                        shuffle_items=True, shuffle_batches=True
                    )
                )
            else:
                train_loader = DataLoader(
                    train_set, num_workers=num_workers, batch_size=self.batch,
                    sampler=BalancedIndexSampler(data_source=train_set,
                                                 shuffle=True)
                )
        else:
            train_loader = DataLoader(train_set, num_workers=num_workers,
                                      batch_size=self.batch, shuffle=True)
        
        return train_loader

    def validation_dataloader(self, dataset, dir, classes, cls_type="classes",
                              quantiles_path=None, num_workers=16):
        if self.criterion_name == "CEL":
            output = "sample"
        elif self.criterion_name == "TML":
            output = "triplet"
        
        val_set = dataset(
            mode="val", dir=dir, classes=classes, cls_type=cls_type,
            quantiles_path=quantiles_path, datasets=self.datasets,
            train_ratio=self.train_ratio, val_ratio=self.val_ratio,
            test_ratio=self.test_ratio, dist_ratio=self.dist_ratio,
            output=output, transform=self.transform_val,
            no_clean_from_attacked=self.no_clean_from_attacked,
            add_clean=self.add_clean,
            add_clean_smart_resize=self.add_clean_smart_resize,
            presets=self.presets, metrics=self.metrics,
            bl_presets=self.bl_presets, bl_metrics=self.bl_metrics,
            crop=self.crop_mode,
        )
        if self.dataset_batching and not self.balanced_sampler:
            val_loader = DataLoader(
                val_set, num_workers=num_workers,
                batch_sampler=DatasetBatchSampler(
                    data_source=val_set, batch_size=self.batch,
                    attacked_datasets=self.datasets
                )
            )
        elif self.crop_mode:
            val_loader = DataLoader(
                val_set, num_workers=num_workers, batch_size=self.batch,
                sampler=CropIndexSampler(data_source=val_set)
            )
        elif self.balanced_sampler:
            if self.dataset_batching:
                val_loader = DataLoader(
                    val_set, num_workers=num_workers,
                    batch_sampler=BalancedDatasetBatchSampler(
                        data_source=val_set, batch_size=self.batch
                    )
                )
            else:
                val_loader = DataLoader(
                    val_set, num_workers=num_workers, batch_size=self.batch,
                    sampler=BalancedIndexSampler(data_source=val_set)
                )
        else:
            val_loader = DataLoader(val_set, num_workers=num_workers,
                                    batch_size=self.batch)
        
        return val_loader

    def test_dataloader(self, dataset, dir, classes, cls_type="classes",
                        quantiles_path=None, num_workers=16):
        if self.criterion_name == "CEL":
            output = "sample"
        elif self.criterion_name == "TML":
            output = "triplet"
        
        test_set = dataset(
            mode="test", dir=dir, classes=classes, cls_type=cls_type,
            quantiles_path=quantiles_path, datasets=self.datasets,
            train_ratio=self.train_ratio,val_ratio=self.val_ratio,
            test_ratio=self.test_ratio, output=output,
            transform=self.transform_val,
            no_clean_from_attacked=self.no_clean_from_attacked,
            presets=self.presets, metrics=self.metrics,
            bl_presets=self.bl_presets, bl_metrics=self.bl_metrics,
        )
        test_loader = DataLoader(test_set, num_workers=num_workers,
                                 batch_size=self.batch)
        
        return test_loader
