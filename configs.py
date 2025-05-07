
#### 2. 配置文件


# 文件路径配置
import os

# 数据路径
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw/')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed/')

# 创建目录
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# 模型训练参数
EMBEDDING_DIM = 64
LATENT_DIM = 100
NUM_EPOCHS = 80
LEARNING_RATE = 0.001
BATCH_SIZE = 64