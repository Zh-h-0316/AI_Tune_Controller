import os
import shutil
import time

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DST_DIR = os.path.join(SRC_DIR, 'Copy_raw')

# 需要同步的文件和文件夹（可根据实际情况调整）
SYNC_LIST = [
    'Adaptive Network.py',
    'AI_main.py',
    'by_source_analyzer.py',
    'check_deployment_env.py',
    'Config_Para.py',
    'data_structures.py',
    'DEPLOYMENT_GUIDE.md',
    'deployment_reference.py',
    'DEPLOYMENT_SOLUTION.md',
    'inference_main.py',
    'LQR_main.py',
    'LQR_ratio.py',
    'model_deployment.py',
    'Model_Design.md',
    'path_generator.py',
    'QUICK_REFERENCE.py',
    'RL_finetune.py',
    'vehicle_model.py',
    'Code_Txt',
    'deploy_tools',
]

def sync_file(src, dst):
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

def sync_all():
    for item in SYNC_LIST:
        src_path = os.path.join(SRC_DIR, item)
        dst_path = os.path.join(DST_DIR, item)
        if os.path.exists(src_path):
            sync_file(src_path, dst_path)

if __name__ == '__main__':
    print('正在同步源代码到 Copy_raw 文件夹...')
    sync_all()
    print('同步完成。')
