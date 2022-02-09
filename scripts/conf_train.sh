set -x
data_path='/workspace/bonito/bonito/data/dna_r9.4.1'

dst_dir='bonito/data/conform'
mkdir ${dst_dir} 
dst_path='${dst_dir}/conf1_b32_e30_2_4'
conf='bonito/models/configs/dna_r9.4.1\@v_conf.toml'
seed=42
lr=2e-4
bs=32
ep=30
log_path='log_conf_2_4.txt'

rm -r ${dst_path}  
CUDA_VISIBLE_DEVICES=1 nohup bonito train ${dst_path} -f --directory ${data_path} --config ${conf}  --lr $lr --epoch $ep --batch $bs --seed ${seed} --no-amp  > ${log_path} & 
