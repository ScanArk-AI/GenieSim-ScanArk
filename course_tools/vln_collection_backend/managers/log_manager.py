import logging
import os
from datetime import datetime

# 默认配置（基础配置，只输出到终端）
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s",
#     handlers=[
#         logging.StreamHandler()  # 终端输出
#     ]
# )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[]
)

logger = logging.getLogger(__name__)

def setup_file_logger(output_dir):
    """配置文件日志处理器，按时间命名日志文件
    
    Args:
        output_dir: 日志输出目录
        
    Returns:
        log_file: 日志文件路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 按时间命名日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"run_{timestamp}.log")
    
    # 创建文件处理器
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    # 获取 root logger：同时写文件与终端（stderr）
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    logger.info(f"✓ 日志文件已配置: {log_file}")
    
    return log_file
