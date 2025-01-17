set -x
data_path='/workspace/bonito/bonito/data/dna_r9.4.1'

dst_dir='bonito/data/conform'
mkdir ${dst_dir} 
dst_path="${dst_dir}/confctc_e5_2_4"
conf='bonito/models/configs/dna_r9.4.1\@v_conf_ctc_v2.toml'
seed=42
lr=1e-4
bs=8
ep=5
log_path='log_conf_ctc_1_4.txt'

rm -r ${dst_path}  
CUDA_VISIBLE_DEVICES=0 nohup bonito train ${dst_path} -f --directory ${data_path} --config ${conf}  --lr $lr --epoch $ep --batch $bs --seed ${seed} --no-amp  > ${log_path} & 
# CUDA_VISIBLE_DEVICES=0 nohup bonito train ${dst_path} -f --directory ${data_path} --config ${conf}  --lr $lr --epoch $ep --batch $bs --seed ${seed}  > ${log_path} & 
