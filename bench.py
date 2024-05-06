import os
import pickle
from generator.generator import GraphGenerator
# 导入加载数据集的函数
from load_dataset import load_dataset, split_data

# 定义数据集所在的目录
dataset_dir = 'data'  # 替换为你的数据集目录

import time
start = time.time()

GeneratorVariant = GraphGenerator
gen_file = f'gen{GeneratorVariant.variant_name()}.pkl'
cosponsors, members, votes = load_dataset(dataset_dir)
gen = GeneratorVariant(proposals=cosponsors, members=members, votes=votes)
with open(os.path.join('./data', gen_file), 'wb') as f:
    pickle.dump(gen, f)

print(f"Time elapsed: {time.time() - start}s")