#python -u main.py --anormly_ratio 1 --num_epochs 3   --batch_size 8  --mode train --dataset MSL  --data_path /data/dkgud111/AnoFormer/dataset/MSL --input_c 55    --output_c 55
python -u main.py --anormly_ratio 1  --num_epochs 10      --batch_size 8     --mode test    --dataset MSL   --data_path /data/dkgud111/AnoFormer/dataset/MSL  --input_c 55    --output_c 55  --pretrained_model 20


