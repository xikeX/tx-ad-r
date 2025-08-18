#!/bin/bash

# 定义输出的 zip 文件名
zip_name="myarchive.zip"

# 定义要压缩的文件列表（直接在这里填写）
files_to_zip=(
    faiss_demo.py
    dataset.py
    my_dataset.py
    embedding_model.py
    infer_class.py
    main.py
    model.py
    info_only.py
    model_rqvae.py
    split_embedding_model.py
)

# 执行压缩
zip -r "$zip_name" "${files_to_zip[@]}"