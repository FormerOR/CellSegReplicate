import os
import json
import random

# 1. 配置
dataset = "MO"
img_dir = os.path.join("data1", dataset, "images")
out_path = os.path.join("data1", dataset, "train_val_test.json")

# 2. 收集所有文件名（不含后缀也行，保持和你后续加载时一致）
all_imgs = sorted(os.listdir(img_dir))  # e.g. ["img001.png", "img002.png", ...]

# 3. 打乱顺序
random.seed(2025)
random.shuffle(all_imgs)

# 只取30张
# all_imgs = all_imgs[:30]  # 如果需要全部数据，可以注释掉这一行

# 4. 划分比例
n = len(all_imgs)
n_train = 22
n_val   = 5

train_list = all_imgs[:n_train]
val_list   = all_imgs[n_train:n_train+n_val]
test_list  = all_imgs[n_train+n_val:]

# 5. 写入 JSON
data = {
    "train": train_list,
    "val":   val_list,
    "test":  test_list
}

with open(out_path, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Saved split to {out_path}  (train={len(train_list)}, val={len(val_list)}, test={len(test_list)})")
