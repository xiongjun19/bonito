set -x
data_path='/workspace/bonito/bonito/data/dna_r9.4.1'

dst_dir='bonito/data/conform'
mkdir ${dst_dir} 
dst_path="${dst_dir}/trans_22_scra_2_4_ln"
conf='bonito/models/configs/dna_r9.4.1\@v_conf_trans22_ft.toml'
seed=42
lr=2e-4
bs=32
ep=20
log_path='log_conf_trans_22.txt'

rm -r ${dst_path}  
CUDA_VISIBLE_DEVICES=1 bonito train ${dst_path} -f --directory ${data_path} --config ${conf}  --lr $lr --epoch $ep --batch $bs --seed ${seed} --no-amp

