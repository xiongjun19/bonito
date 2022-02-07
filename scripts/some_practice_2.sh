set -x
dst_path='bonito/data/test_trans_b48_e20_4_4'
data_path='/workspace/bonito/bonito/data/dna_r9.4.1'
conf='bonito/models/configs/dna_r9.4.1\@v_trans_1.toml'
seed=42
lr=3e-4
bs=48
ep=20
log_path='log_new_48.txt'

rm -r ${dst_path}  
CUDA_VISIBLE_DEVICES=0 nohup bonito train ${dst_path} -f --directory ${data_path} --config ${conf}  --lr $lr --epoch $ep --batch $bs --seed ${seed} > ${log_path} & 
