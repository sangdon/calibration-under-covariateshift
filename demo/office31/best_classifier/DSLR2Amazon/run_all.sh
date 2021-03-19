screen -dm bash -c "CUDA_VISIBLE_DEVICES=0 python3 exp_Temp.py > output_exp_Temp"
screen -dm bash -c "CUDA_VISIBLE_DEVICES=1 python3 exp_Temp_IW.py > output_exp_Temp_IW"
screen -dm bash -c "CUDA_VISIBLE_DEVICES=2 python3 exp_Temp_FL.py > output_exp_Temp_FL"
screen -dm bash -c "CUDA_VISIBLE_DEVICES=3 python3 exp_Temp_FL_IW.py > output_exp_Temp_FL_IW"
