#!/bin/bash
# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

pip install hnswlib
# 要解压的 zip 文件名（修改为你自己的文件名）
zip_file="myarchive.zip"

# 解压到当前目录，并自动覆盖文件
unzip -o "$zip_file"

# write your code below
python -u main.py --train_name=v3 --use_lr_scheduler_in_downstream 

python -u infer_main.py --train_name=v3
