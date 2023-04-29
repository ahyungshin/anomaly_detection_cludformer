python -u main.py --anormly_ratio 1 \
                    --num_epochs 10  \
                    --cluster_temporal 20 \
                    --cluster_channel 4 \
                    --batch_size 8  \
                    --mode train \
                    --dataset MSL  \
                    --data_path /path/to/dataset/ \
                    --input_c 55    \
                    --output_c 55
python -u main.py --anormly_ratio 1  \
                    --num_epochs 10      \
                    --cluster_temporal 20 \
                    --cluster_channel 4 \
                    --batch_size 8     \
                    --mode test    \
                    --dataset MSL   \
                    --data_path /path/to/dataset/  \
                    --input_c 55    \
                    --output_c 55  \
                    --pretrained_model 20


