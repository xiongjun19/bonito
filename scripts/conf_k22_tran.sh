set -x
data_path='/workspace/bonito/bonito/data/dna_r9.4.1'

dst_dir='bonito/data/conform'
mkdir ${dst_dir} 
dst_path="${dst_dir}/conf_k22_e10_1_4"

conf='bonito/models/configs/conf_res_k22.toml'
seed=42
lr=4e-3
bs=32
ep=2
log_path='log_conf_trans_22.txt'

# rm -r ${dst_path}  
# CUDA_VISIBLE_DEVICES=0 nohup bonito train ${dst_path} -f --directory ${data_path} --config ${conf}  --lr $lr --epoch $ep --batch $bs --seed ${seed} --no-amp  > ${log_path} & 
CUDA_VISIBLE_DEVICES=0 bonito train ${dst_path} -f --directory ${data_path} --config ${conf}  --lr $lr --epoch $ep --batch $bs --seed ${seed} --no-amp

conf='bonito/models/configs/conf_res_k22_ft.toml'
seed=42
lr=1e-4
bs=16
ep=10
log_path='log_conf_trans_22.txt'


CUDA_VISIBLE_DEVICES=0 bonito train ${dst_path} -f --directory ${data_path} --config ${conf}  --lr $lr --epoch $ep --batch $bs --seed ${seed} --no-amp
