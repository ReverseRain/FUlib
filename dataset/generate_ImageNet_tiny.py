# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import os
import sys
import random
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.model_selection import train_test_split
from utils.dataset_utils import check

random.seed(1)
np.random.seed(1)
num_clients = 100
dir_path = "ImageNet_tiny/"

# Tiny ImageNet: 200 classes, each with 500 train images (64x64) and 50 test images
TINY_IMAGENET_NUM_CLASSES = 200
TINY_IMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINY_IMAGENET_STD = (0.2302, 0.2265, 0.2262)


def load_tiny_imagenet_raw(data_dir):
    """
    Load Tiny ImageNet from a local directory structure.
    
    Expected structure:
        data_dir/
            train/          # each subfolder is a class, contains JPEG images
                n01443537/
                    images/
                        n01443537_0.JPEG
                        ...
                ...
            val/            # validation: val_annotations.txt maps images to labels
                images/
                    val_0.JPEG
                    ...
                val_annotations.txt
            wnids.txt       # list of all 200 class names (synset IDs)
            words.txt       # mapping from synset ID to human-readable label
    
    If the data does not exist locally, the function will download it automatically.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    wnids_file = os.path.join(data_dir, "wnids.txt")

    images = []
    labels = []

    # ---- Check if data exists, if not, download ----
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"Tiny ImageNet data not found at {data_dir}. Downloading...")
        _download_tiny_imagenet(data_dir)
        print("Download complete.")

    # ---- Load class names (synset IDs) ----
    with open(wnids_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    # ---- Load training data ----
    print("Loading training data...")
    for class_name in class_names:
        class_train_dir = os.path.join(train_dir, class_name, "images")
        if not os.path.exists(class_train_dir):
            # Some versions of Tiny ImageNet store images directly in class folder
            class_train_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_train_dir):
            for img_name in os.listdir(class_train_dir):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png', '.JPEG')):
                    img_path = os.path.join(class_train_dir, img_name)
                    images.append(img_path)
                    labels.append(class_to_idx[class_name])

    # ---- Load validation data ----
    print("Loading validation data...")
    val_images_dir = os.path.join(val_dir, "images")
    val_annotations_file = os.path.join(val_dir, "val_annotations.txt")

    if os.path.exists(val_annotations_file):
        with open(val_annotations_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name, class_name = parts[0], parts[1]
                    if class_name in class_to_idx:
                        img_path = os.path.join(val_images_dir, img_name)
                        if os.path.exists(img_path):
                            images.append(img_path)
                            labels.append(class_to_idx[class_name])
    else:
        # Fallback: load val images as ImageFolder if structure is similar to train
        if os.path.exists(val_dir):
            val_dataset = ImageFolder(val_dir)
            for img_path, label in val_dataset.samples:
                images.append(img_path)
                labels.append(label)

    print(f"Loaded {len(images)} images with {len(set(labels))} classes.")
    return images, labels, len(class_names)


def _download_tiny_imagenet(data_dir):
    """Download and extract Tiny ImageNet dataset."""
    import urllib.request
    import zipfile
    import shutil

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    extract_dir = os.path.join(data_dir, "temp_extract")

    os.makedirs(data_dir, exist_ok=True)

    # Download
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download finished. Extracting...")

    # Extract
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Move contents to expected structure
    extracted_root = os.path.join(extract_dir, "tiny-imagenet-200")
    if os.path.exists(extracted_root):
        for item in os.listdir(extracted_root):
            src = os.path.join(extracted_root, item)
            dst = os.path.join(data_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)

    # Cleanup
    shutil.rmtree(extract_dir, ignore_errors=True)
    os.remove(zip_path)
    print("Extraction complete.")


def read_image_safe(img_path, transform):
    """Read a single image and apply transform, return numpy array or None on failure."""
    try:
        with Image.open(img_path).convert('RGB') as img:
            img_tensor = transform(img)
            return img_tensor.numpy()
    except Exception as e:
        print(f"Warning: Could not read {img_path}: {e}")
        return None


def read_client_images(client_idxs, image_paths, labels, transform):
    """Read images for a specific set of indices (one client's data)."""
    X_client = []
    y_client = []
    for idx in client_idxs:
        img_array = read_image_safe(image_paths[idx], transform)
        if img_array is not None:
            X_client.append(img_array)
            y_client.append(labels[idx])
    return np.array(X_client), np.array(y_client)


def _allocate_indices(labels, num_clients, num_classes, niid, balance, partition, class_per_client=None):
    """
    Allocate data indices to clients using the same logic as separate_data
    from dataset_utils.py, but operating only on label arrays (no image data).
    
    Returns:
        client_idxs: list of lists, client_idxs[client] = [idx1, idx2, ...]
    """
    least_samples = 10
    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(labels)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[labels == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            if len(selected_clients) == 0:
                break
            selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
                                                num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        min_size = 0
        K = num_classes
        N = len(labels)
        alpha = 0.8

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(f'Client data size does not meet the minimum requirement {least_samples}. '
                      f'Trying again ({try_cnt}-th time).')
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(labels == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]

    else:
        raise NotImplementedError(f"Partition '{partition}' not implemented in _allocate_indices.")

    client_idxs = [np.array(dataidx_map[client], dtype=int) for client in range(num_clients)]
    return client_idxs


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    proxy_path = dir_path + "proxy/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # Get Tiny ImageNet data: only load paths and labels (no image data into memory)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD)
    ])

    raw_data_dir = dir_path + "rawdata"
    image_paths, label_list, num_classes = load_tiny_imagenet_raw(raw_data_dir)
    label_list = np.array(label_list)
    print(f'Number of classes: {num_classes}')
    print(f'Total images: {len(image_paths)}')

    # ---- Memory-efficient approach: allocate by indices, load per client ----
    print("Allocating data to clients by indices...")
    client_idxs = _allocate_indices(label_list, num_clients, num_classes,
                                    niid, balance, partition, class_per_client=20)

    # Read and save data per client
    print("Reading and saving data per client...")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(proxy_path, exist_ok=True)

    statistic = [[] for _ in range(num_clients)]

    for client in range(num_clients):
        print(f"Processing client {client+1}/{num_clients}...")
        idxs = client_idxs[client]

        # Read this client's images (only a subset at a time)
        X_client, y_client = read_client_images(idxs, image_paths, label_list, transform)

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_client, y_client, train_size=0.75, shuffle=True)

        train_data = {'x': X_train, 'y': y_train}
        test_data = {'x': X_test, 'y': y_test}

        # Save immediately to disk (free memory)
        with open(train_path + str(client) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_data)
        with open(test_path + str(client) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_data)

        # Track statistics
        for i in np.unique(y_client):
            statistic[client].append((int(i), int(sum(y_client == i))))

        print(f"  Client {client}: {len(X_client)} samples, labels: {np.unique(y_client)}")

        del X_client, y_client, X_train, X_test, y_train, y_test, train_data, test_data

    # Save config
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'balance': balance,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
        'alpha': 0.8,
        'batch_size': 10,
    }

    # Save proxy data (sample subset of paths, not images)
    proxy_sample_paths = random.sample(image_paths, min(1000, len(image_paths)))
    proxy_images = []
    for p in proxy_sample_paths:
        img = read_image_safe(p, transform)
        if img is not None:
            proxy_images.append(img)
    proxy_data = np.array(proxy_images)

    with open(config_path, 'w') as f:
        import ujson
        ujson.dump(config, f)
    np.savez_compressed(proxy_path + "proxy.npz", data=proxy_data)
    print("Finish generating dataset.\n")


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    dir_path = "ImageNet_tiny_noniid_client100/"

    generate_dataset(dir_path, num_clients, niid, balance, partition)
