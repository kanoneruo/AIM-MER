import argparse
from scipy.io import loadmat
import h5py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from dataset.EVdataset import EVdataset
from model.model_eeg import AAA
from utils.config_eeg import config
from utils.seed import set_seed
from trainer.trainer_eeg import Trainer
from utils.save import save_results_to_excel

def main():
    set_seed(config["seed"])
    dataset = h5py.File(config['eeg_data_path'], 'r')
    one_data2 = dataset["DE_features"]
    all_bands = []
    all_face_data = []

    # 读取 face .mat
    mat_files = sorted([f for f in os.listdir(config['face_data_path']) if f.endswith('.mat')])
    for file_name in mat_files:
        file_path = os.path.join(config['face_data_path'], file_name)
        mat_dict = loadmat(file_path)
        data_key = [k for k in mat_dict.keys() if not k.startswith("__")][0]
        data = mat_dict[data_key]  # (60, 60, 256)
        all_face_data.append(data)
    face_data = np.stack(all_face_data, axis=0)                 # [N,60,60,256]
    face_data = torch.tensor(face_data, dtype=torch.float32)

    # 读取 EEG 各频带
    for key in one_data2.keys():
        data = one_data2[key][()]                               # [60, 32, 1280]
        data = np.transpose(data, (2, 1, 0))                    # -> [1280, 32, 60]
        all_bands.append(data)
    stacked = np.stack(all_bands, axis=1)                       # -> [1280, 5, 32, 60]
    stacked = torch.tensor(stacked, dtype=torch.float32)

    # 统一截取前 880
    eeg_data = stacked[:880, :, :, :]
    face_data = face_data[:880, :, :, :]
    labels = dataset["V_labels"]
    labels = np.array(labels).transpose(1, 0)[:880, :]
    labels = torch.tensor(labels, dtype=torch.long).squeeze(1)

    print(eeg_data.shape)   # [880,5,32,60]
    print(face_data.shape)  # [880,60,60,256]
    print(labels.shape)     # [880]

    # ========= 严格按截图：加噪 + cat（翻倍） =========
    # 说明：为保持模态长度一致，Face 也做同样处理（截图只演示 EEG，但我们两边都翻倍）
    augmented_eeg  = eeg_data.clone()
    augmented_face = face_data.clone()
    augmented_y    = labels.clone()

    for _ in range(1):  # 做一轮 -> ×2；想更多就把 1 改成 2、3...
        noise_eeg  = torch.randn_like(eeg_data)  * 0.01  # 少量高斯噪声
        noise_face = torch.randn_like(face_data) * 0.01  # 同步给 Face 加噪，避免长度不一致
        augmented_eeg  = torch.cat((augmented_eeg,  eeg_data  + noise_eeg),  dim=0)
        augmented_face = torch.cat((augmented_face, face_data + noise_face), dim=0)
        augmented_y    = torch.cat((augmented_y,    labels),                 dim=0)

    eeg_data  = augmented_eeg
    face_data = augmented_face
    labels    = augmented_y

    print("EEG after aug :", eeg_data.shape)   # [1760,5,32,60]
    print("Face after aug:", face_data.shape)  # [1760,60,60,256]
    print("y   after aug :", labels.shape)     # [1760]
    # ========= 结束：严格按图实现 =========

    epoch_list = []
    best_test_acc_list = []
    best_test_acc0_list = []
    best_test_acc1_list = []
    best_test_sen_list = []
    best_test_spe_list = []
    best_test_f1_list = []
    best_test_pre_list = []
    best_test_rec_list = []

    kf = StratifiedKFold(n_splits=config['kfold'], shuffle=True, random_state=config['seed'])
    for fold, (train_idx, val_idx) in enumerate(kf.split(eeg_data, labels)):
        train_dataset = EVdataset(eeg_data[train_idx], face_data[train_idx], labels[train_idx])
        test_dataset  = EVdataset(eeg_data[val_idx],  face_data[val_idx],  labels[val_idx])
        train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        test_dataloader  = DataLoader(test_dataset,  batch_size=config["batch_size"], shuffle=False)

        model = AAA()
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, config["lr_decay_step"], config["lr_decay_ratio"])

        trainer = Trainer(model, train_dataloader, test_dataloader, optimizer, scheduler,
                          config["device"], config["log_to_file"], config, fold+1)
        metrics = trainer.train(num_epochs=config['epochs'])

        epoch_list.append(metrics['best_epoch'])
        best_test_acc_list.append(metrics['best_accuracy'])
        best_test_acc0_list.append(metrics['best_accuracy0'])
        best_test_acc1_list.append(metrics['best_accuracy1'])
        best_test_sen_list.append(metrics['sensitivity'])
        best_test_spe_list.append(metrics['specificity'])
        best_test_f1_list.append(metrics['f1'])
        best_test_pre_list.append(metrics['precision'])
        best_test_rec_list.append(metrics['recall'])

    for idx, acc in enumerate(best_test_acc_list, 1):
        print(f"Fold {idx}: acc:{best_test_acc_list[idx - 1]:.4f}")
        print(f"Fold {idx}: acc0:{best_test_acc0_list[idx - 1]:.4f}")
        print(f"Fold {idx}: acc1:{best_test_acc1_list[idx - 1]:.4f}")
        print(f"Fold {idx}: sen:{best_test_sen_list[idx - 1]:.4f}")
        print(f"Fold {idx}: spe:{best_test_spe_list[idx - 1]:.4f}")
        print(f"Fold {idx}: f1:{best_test_f1_list[idx - 1]:.4f}")
        print(f"Fold {idx}: pre:{best_test_pre_list[idx - 1]:.4f}")
        print(f"Fold {idx}: rec:{best_test_rec_list[idx - 1]:.4f}")

    print(f"\nK-Fold CV Avg Accuracy: {np.mean(best_test_acc_list):.2f} ± {np.std(best_test_acc_list):.2f} ")
    for idx, epoo in enumerate(epoch_list, 1):
        print(f"Fold {idx}: {epoo:.2f}")

    path = os.path.join(config["result_excel_path"], config["experiment_name"],
                        f"result_{config['experiment_name']}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_results_to_excel(path, best_test_acc_list, best_test_acc0_list,
                          best_test_acc1_list, best_test_sen_list, best_test_spe_list,
                          best_test_f1_list, best_test_pre_list, best_test_rec_list, epoch_list)

if __name__ == "__main__":
    main()
