import os
import numpy as np
import h5py
import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from dataset.EVdataset import EVdataset
from model.model_eye import AAA
from utils.config_eye import config
from utils.seed import set_seed
from trainer.trainer_eye import Trainer
from utils.save import save_results_to_excel

def main():
    # =====================
    # 0. 固定随机种子
    # =====================
    set_seed(config["seed"])

    # =====================
    # 1. 读 EEG（5 个频带）
    # =====================
    dataset = h5py.File(config['eeg_data_path'], 'r')

    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    all_bands = []

    for b in bands:
        x = dataset[f"DE_features/{b}"][()]     # (10, 62, 1080)
        x = np.transpose(x, (2, 1, 0))          # -> (1080, 62, 10)
        all_bands.append(x)

    eeg_data = np.stack(all_bands, axis=1)      # (1080, 5, 62, 10)
    eeg_data = torch.tensor(eeg_data, dtype=torch.float32)

    # =====================
    # 2. EEG -> (N,5,32,60)
    #    完全照你给的逻辑
    # =====================
    N = eeg_data.shape[0]

    # 10 -> 60（1D 插值）
    eeg_data = eeg_data.view(N * 5, 62, 10)     # (N*5, 62, 10)
    eeg_data = F.interpolate(
        eeg_data,
        size=60,
        mode='linear',
        align_corners=False
    )                                           # (N*5, 62, 60)

    # 62 -> 32（2D 插值）
    eeg_data = eeg_data.unsqueeze(1)            # (N*5, 1, 62, 60)
    eeg_data = F.interpolate(
        eeg_data,
        size=(32, 60),
        mode='bilinear',
        align_corners=False
    )                                           # (N*5, 1, 32, 60)

    eeg_data = eeg_data.squeeze(1).view(N, 5, 32, 60)  # (N,5,32,60)

    # =====================
    # 3. 读 Eye
    # =====================
    with h5py.File(config["eye_data_path"], "r") as f:
        eye = f["EYE_data"][()]                  # (10, 31, 1080)

    eye_data = np.transpose(eye, (2, 1, 0))      # -> (1080, 31, 10)
    eye_data = torch.tensor(eye_data, dtype=torch.float32)

    # =====================
    # 4. 读 labels（四分类）
    # =====================
    labels = np.array(dataset["labels"][()]).reshape(-1).astype(np.int64)
    labels = torch.tensor(labels, dtype=torch.long)

    # =====================
    # 5. 可选：截前 880（与你原代码一致）
    # =====================
    take_n = int(config.get("take_first", 880))
    if take_n > 0 and take_n < eeg_data.shape[0]:
        eeg_data = eeg_data[:take_n]
        eye_data = eye_data[:take_n]
        labels   = labels[:take_n]

    print("EEG:", eeg_data.shape)   # [N,5,32,60]
    print("EYE:", eye_data.shape)   # [N,31,10]
    print("y  :", labels.shape, labels.min().item(), labels.max().item())

    # =====================
    # 6. 数据增强（与你原逻辑一致）
    # =====================
    augmented_eeg = eeg_data.clone()
    augmented_eye = eye_data.clone()
    augmented_y   = labels.clone()

    for _ in range(1):  # 一轮 -> 翻倍
        noise_eeg = torch.randn_like(eeg_data) * 0.01
        noise_eye = torch.randn_like(eye_data) * 0.01

        augmented_eeg = torch.cat([augmented_eeg, eeg_data + noise_eeg], dim=0)
        augmented_eye = torch.cat([augmented_eye, eye_data + noise_eye], dim=0)
        augmented_y   = torch.cat([augmented_y, labels], dim=0)

    eeg_data = augmented_eeg
    eye_data = augmented_eye
    labels   = augmented_y

    print("EEG after aug :", eeg_data.shape)
    print("EYE after aug :", eye_data.shape)
    print("y   after aug :", labels.shape)

    # =====================
    # 7. K-Fold 训练
    # =====================
    epoch_list = []
    best_test_acc_list = []
    best_test_acc0_list = []
    best_test_acc1_list = []
    best_test_sen_list = []
    best_test_spe_list = []
    best_test_f1_list = []
    best_test_pre_list = []
    best_test_rec_list = []

    kf = StratifiedKFold(
        n_splits=config["kfold"],
        shuffle=True,
        random_state=config["seed"]
    )

    y_np = labels.cpu().numpy()
    dummy_X = np.zeros_like(y_np)   # 防止 sklearn 吃大 tensor

    for fold, (train_idx, val_idx) in enumerate(kf.split(dummy_X, y_np), start=1):
        train_dataset = EVdataset(
            eeg_data[train_idx],
            eye_data[train_idx],
            labels[train_idx]
        )
        test_dataset = EVdataset(
            eeg_data[val_idx],
            eye_data[val_idx],
            labels[val_idx]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False
        )

        model = AAA()
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=0.9,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["lr_decay_step"],
            gamma=config["lr_decay_ratio"]
        )

        trainer = Trainer(
            model,
            train_loader,
            test_loader,
            optimizer,
            scheduler,
            config["device"],
            config["log_to_file"],
            config,
            fold
        )

        metrics = trainer.train(num_epochs=config["epochs"])

        epoch_list.append(metrics["best_epoch"])
        best_test_acc_list.append(metrics["best_accuracy"])
        best_test_acc0_list.append(metrics["best_accuracy0"])
        best_test_acc1_list.append(metrics["best_accuracy1"])
        best_test_sen_list.append(metrics["sensitivity"])
        best_test_spe_list.append(metrics["specificity"])
        best_test_f1_list.append(metrics["f1"])
        best_test_pre_list.append(metrics["precision"])
        best_test_rec_list.append(metrics["recall"])

    for idx, acc in enumerate(best_test_acc_list, 1):
        print(f"Fold {idx}: acc:{best_test_acc_list[idx - 1]:.4f}")
        print(f"Fold {idx}: acc0:{best_test_acc0_list[idx - 1]:.4f}")
        print(f"Fold {idx}: acc1:{best_test_acc1_list[idx - 1]:.4f}")
        print(f"Fold {idx}: sen:{best_test_sen_list[idx - 1]:.4f}")
        print(f"Fold {idx}: spe:{best_test_spe_list[idx - 1]:.4f}")
        print(f"Fold {idx}: f1:{best_test_f1_list[idx - 1]:.4f}")
        print(f"Fold {idx}: pre:{best_test_pre_list[idx - 1]:.4f}")
        print(f"Fold {idx}: rec:{best_test_rec_list[idx - 1]:.4f}")

    print(
        f"\nK-Fold CV Avg Accuracy: "
        f"{np.mean(best_test_acc_list):.2f} ± {np.std(best_test_acc_list):.2f}"
    )

    path = os.path.join(
        config["result_excel_path"],
        config["experiment_name"],
        f"result_{config['experiment_name']}.csv"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_results_to_excel(
        path,
        best_test_acc_list,
        best_test_acc0_list,
        best_test_acc1_list,
        best_test_sen_list,
        best_test_spe_list,
        best_test_f1_list,
        best_test_pre_list,
        best_test_rec_list,
        epoch_list
    )


if __name__ == "__main__":
    main()
