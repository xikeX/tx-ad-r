#!/bin/bash
# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

pip install faiss-cpu
# 要解压的 zip 文件名（修改为你自己的文件名）
zip_file="myarchive.zip"

# 解压到当前目录，并自动覆盖文件
unzip -o "$zip_file"

# write your code below
python -u main.py --use_lr_scheduler_in_downstream --verbose
