export DISPLAY=:0.0
export CUDA_VISIBLE_DEVICES=9

## change data folder in train.py
## change tasks in /home/ishika/peract_dir/RVT/rvt/utils/rvt_utils.py

python eval.py \
    --model-folder runs/rvt_bs_12_NW_12  \
    --eval-datafolder /home/ishika/peract_dir/peract/data/test_variations_final \
    --tasks all \
    --eval-episodes 2 \
    --log-name test/variations \
    --device 0 \
    --headless \
    --model-name model_14.pth \
    --save-video \
    >> /home/ishika/peract_dir/RVT/rvt/runs/rvt_bs_12_NW_12/eval/logs.txt



