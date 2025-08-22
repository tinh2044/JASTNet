import json
import pickle

import numpy as np
import torch
import os
from augmentations import (
    rotate_keypoints,
    noise_injection,
    clip_frame,
    time_warp_uniform,
    flip_keypoints,
)


class Datasets(torch.utils.data.Dataset):
    def __init__(self, root, split, shuffle=True, augment=False, keypoints_index=[]):
        self.root = root
        self.split = split
        self.augment = augment
        self.keypoints_index = keypoints_index
        with open(f"{self.root}/label2id.json", "r") as f:
            self.label2id = json.load(f)

        with open(f"{self.root}/id2label.json", "r") as f:
            self.id2label = json.load(f)

        self.data_dir = f"{root}/{split}"

        self.list_key = [
            f"{self.data_dir}/{x}/{y}"
            for x in os.listdir(self.data_dir)
            for y in os.listdir(f"{self.data_dir}/{x}")
        ]

        if shuffle:
            np.random.shuffle(self.list_key)

    def __getitem__(self, i):
        key = self.list_key[i]
        with open(key, "rb") as f:
            sample = pickle.load(f)

        keypoints = np.array(sample["keypoints"])[:, self.keypoints_index, :2]
        assert sample["class"] == key.split("/")[-2], (
            f"{sample['class']} != {key.split('/')[-2]}"
        )
        label = self.label2id[sample["class"]]

        keypoints = np.clip(keypoints, 0, 1)
        if keypoints.shape[0] > 64:
            keypoints = clip_frame(keypoints, 64, True)

        if self.augment and np.random.uniform(0, 1) < 0.4:
            keypoints = self.apply_augment(keypoints)

        # keypoints = self.normalize_keypoints(keypoints)

        keypoints = torch.from_numpy(keypoints).float()
        label = torch.tensor(label, dtype=torch.long)

        return keypoints, label

    def apply_augment(self, keypoints):
        aug = False
        while not aug:
            if np.random.uniform(0, 1) < 0.5:
                angle = np.random.uniform(-15, 15)
                keypoints = rotate_keypoints(keypoints, (0, 0), angle)
                aug = True
            if np.random.uniform(0, 1) < 0.5:
                random_noise = np.random.uniform(0.01, 0.2)
                keypoints = noise_injection(keypoints, random_noise)
                aug = True
            if np.random.uniform(0, 1) < 0.5:
                n_f = keypoints.shape[0]
                tgt = np.random.randint(n_f // 2, n_f)
                is_uniform = np.random.uniform(0, 1) < 0.5
                keypoints = clip_frame(keypoints, tgt, is_uniform)
                aug = True
            if np.random.uniform(0, 1) < 0.5:
                speed = np.random.uniform(1.1, 1.5)
                keypoints = time_warp_uniform(keypoints, speed)
                aug = True

            if np.random.uniform(0, 1) < 0.5:
                keypoints = flip_keypoints(keypoints)
                aug = True

        return keypoints

    def query_class(self, class_name, _max=5):
        key_querys = [self.list_key.index(x) for x in self.list_key if class_name in x][
            :_max
        ]
        return [self[i] for i in key_querys]

    def __len__(self):
        return len(self.list_key)

    def padding_keypoint(self, keypoint, max_len):
        T, K, C = keypoint.shape
        padding = torch.zeros((max_len - T, K, C))
        kp_padd = torch.cat((keypoint, padding), dim=0)

        return kp_padd

    def data_collator(self, batch):
        keypoints = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        lengths = [x.shape[0] for x in keypoints]
        max_len = max(lengths)

        attention_masks = []
        keypoint_paddings = []
        for keypoint in keypoints:
            T, K, C = keypoint.shape
            if T < max_len:
                mask = torch.cat((torch.ones(T), torch.zeros(max_len - T)), dim=-1)
                kp_padd = self.padding_keypoint(keypoint, max_len)
            else:
                mask = torch.ones(T)
                kp_padd = keypoint
            attention_masks.append(mask)
            keypoint_paddings.append(kp_padd)

        keypoints = torch.stack(keypoint_paddings)
        attention_mask = torch.stack(attention_masks)

        labels = torch.stack(labels)

        return {
            "keypoints": keypoints,
            "attention_mask": attention_mask,
            "labels": labels,
        }


if __name__ == "__main__":
    x = np.random.randint(0, 3)
    print(type(x))
