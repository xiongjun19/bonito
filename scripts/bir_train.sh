set -x
data_path='/workspace/bonito/bonito/data/dna_r9.4.1'

mkdir 'bonito/data/pes_bir'
dst_path='bonito/data/pes_bir/pes_bir_b32_e20_2_4'
conf='bonito/models/configs/dna_r9.4.1\@v_trans_pe_bidir.toml'
seed=42
lr=2e-4
bs=32
ep=20
log_path='log_bidir_2_4.txt'

rm -r ${dst_path}  
CUDA_VISIBLE_DEVICES=0 nohup bonito train ${dst_path} -f --directory ${data_path} --config ${conf}  --lr $lr --epoch $ep --batch $bs --seed ${seed} > ${log_path} & 
