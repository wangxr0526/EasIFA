#!/bin/bash

# 定义输入和输出文件夹
input_directory=$1
output_directory=$2

# 检查输出文件夹是否存在，如果不存在则创建
if [ ! -d "$output_directory" ]; then
  mkdir -p "$output_directory"
fi

# 遍历输入文件夹中的所有 PDB 文件
for pdb_file in "$input_directory"/*.pdb; do
  # 获取文件名（不包含路径）
  file_name=$(basename "$pdb_file")
  
  # 定义输出文件的完整路径
  output_file="$output_directory/$file_name"

  # 使用 OpenBabel 优化 PDB 文件并添加氢原子
  obabel -ipdb "$pdb_file" -opdb -O "$output_file" -h

  # 反馈操作结果
  echo "Processed $file_name"
done

echo "All PDB files have been optimized and saved to $output_directory."
