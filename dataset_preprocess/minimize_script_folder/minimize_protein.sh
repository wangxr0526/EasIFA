#!/bin/bash

# 检查输入参数
if [[ "$#" -lt 2 ]]; then
    echo "Usage: $0 input_pdb output_pdb [AMBERHOME]"
    exit 1
fi

input_pdb=$1
output_folder=$2

if [[ -n "$3"  ]]; then
    AMBERHOME="$3"
elif [[ -z "$AMBERHOME" ]]; then
    AMBERHOME="/home/xiaoruiwang/software/amber20"
fi

# 输入的 PDB 文件

if [[ ! -d "$output_folder" ]]; then
    echo "Directory does not exist. Creating directory..."
    mkdir -p "$output_folder"
fi

# 提取 PDB 文件的基本名称（不含扩展名）
filename="$(basename "$input_pdb")"
dirpath="$(dirname "$input_pdb")"
base_name="${filename%.pdb}"

# 中间文件输出位置
tmp_path=/tmp

# 定义输出文件名的前缀
output_prefix="${base_name}_minimized"

#step 1: 使用 tleap 生成拓扑和坐标文件
tleap_input="${tmp_path}/${output_prefix}_tleap.in"
cat > $tleap_input <<EOF
source leaprc.protein.ff14SB
protein = loadpdb $input_pdb
saveamberparm protein ${tmp_path}/${output_prefix}.prmtop ${tmp_path}/${output_prefix}.inpcrd
quit
EOF

${AMBERHOME}/bin/tleap -f $tleap_input > ${output_folder}/${output_prefix}_tleap.log

# Step 2: 能量最小化
min_input="${tmp_path}/${output_prefix}_min.in"
cat > $min_input <<EOF
dry protein minimization
&cntrl
   imin=1, maxcyc=1000, ncyc=500,
   ntb=0, igb=1, 
   cut=999.0,
/
EOF

# 使用 pmemd.cuda 进行 GPU 加速计算
${AMBERHOME}/bin/pmemd.cuda -O -i $min_input -o ${tmp_path}/${output_prefix}.out -p ${tmp_path}/${output_prefix}.prmtop -c ${tmp_path}/${output_prefix}.inpcrd -r ${tmp_path}/${output_prefix}.rst

# Step 3: 输出优化后的 PDB 结构
${AMBERHOME}/bin/ambpdb -p ${tmp_path}/${output_prefix}.prmtop -c ${tmp_path}/${output_prefix}.rst > ${output_folder}/${output_prefix}.pdb

echo "Minimization complete. Optimized structure is in ${output_folder}/${output_prefix}.pdb"

