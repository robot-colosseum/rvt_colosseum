export DISPLAY=:0.0

export eval_dir=eval_variations_final
export DEMO_PATH=/home/ishika/peract_dir/peract/data/test_variations_final
export eval_episodes=25


## change data folder in train.py
## change tasks in /home/ishika/peract_dir/RVT/rvt/utils/rvt_utils.py

export device=2
counter=0
counter_inisde=0

for task_list in $(ls /home/ishika/peract_dir/peract/data/test_variations_final/); do

        if [[ $counter -ge $1 ]] && \
            [[ $counter -le $2 ]] && \
            [[ $counter_inisde -le 50 ]] && \
            [[ $task_list == *_17 ]] # ||  $task_list == scoop_with_spatula_6 ||  $task_list == setup_chess_6 ||  $task_list == close_box_6 ]] # || $task_list == setup_chess_6 ]] #|| $task_list == setup_chess_6 || $task_list == place_wine_at_rack_location_6 || $task_list == empty_dishwasher_6 || $task_list == get_ice_from_fridge_7 ]] # || $task_list == empty_dishwasher_7 || $task_list == hockey_7 || $task_list == place_wine_at_rack_location_7 || $task_list == insert_onto_square_peg_7 || $task_list == scoop_with_spatula_7 ]]
        then
            if [ $(( counter_inisde % 3 )) == 0 ]
            then
                device=$((device-1)) 
            fi

            ##
            # export device=4

            echo $device:rvt:$task_list:$counter
            counter_inisde=$(( counter_inisde + 1 ))

            

            CUDA_VISIBLE_DEVICES=$device python eval.py \
                --model-folder runs/rvt_bs_12_NW_12  \
                --eval-datafolder $DEMO_PATH \
                --tasks $task_list \
                --eval-episodes $eval_episodes \
                --log-name test/$eval_dir \
                --device 0 \
                --headless \
                --model-name model_14.pth \
                --save-video \
                >> /home/ishika/peract_dir/RVT/rvt/runs/rvt_bs_12_NW_12/eval/logs/rvt_${task_list}_final.txt &
        fi
    counter=$((counter+1))
done





