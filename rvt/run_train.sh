export DISPLAY=:0.0
export CUDA_VISIBLE_DEVICES=0

## change data folder in train.py
## change tasks in /home/ishika/peract_dir/RVT/rvt/utils/rvt_utils.py

# python train.py \
#     --exp_cfg_opts "bs 12 num_workers 12" \
#     --exp_cfg_path configs/all.yaml \
#     --device 0,1


python train.py \
    --exp_cfg_opts "bs 4 num_workers 4" \
    --exp_cfg_path configs/all.yaml \
    --device 0



