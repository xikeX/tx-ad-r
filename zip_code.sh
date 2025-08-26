#!/bin/bash

# 定义输出的 zip 文件名
zip_name="myarchive.zip"

# 定义要压缩的文件列表（直接在这里填写）
files_to_zip=(
baseline_model_v1.py
baseline_model_v2_QK_norm.py
baseline_model_v3_infonce.py
baseline_model_v3_infonce2323.py
baseline_model_v4_ctcvr_v2.py
baseline_model_v4_ctcvr.py
baseline_model_v5_senet.py
baseline_model.py
dataset.py
embedding_model.py
faiss_demo.py
infer_class_v1.py
infer_class.py
infer_only.py
main_v1.py
main_v2.py
main_v3.py
main_v4.py
main.py
model_rqvae.py
model_x.py
my_dataset_v1_aug.py
my_dataset_v1.py
my_dataset.py
base/*
split_embedding_model.py
infer_main.py
main_v3/*.py
main_v3/*.sh

main_v7.py
hstu.py
baseline_model_v7_hstu.py
)

# 执行压缩
zip -r "$zip_name" "${files_to_zip[@]}"