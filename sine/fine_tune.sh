python main.py \
    --base ./v1-finetune_picture.yaml \
    -t --actual_resume './models/pretrained.ckpt' \
    --obj "dog sitting in the library" --desc "near focus camera" \
    -n "NearFocus_Train" --gpus 0, --logdir ./logs \
    --data_root "./data/near_focus/train.png" \
    --reg_data_root "./data/near_focus/train.png"
    
python main.py \
    --base ./v1-finetune_picture.yaml \
    -t --actual_resume './models/pretrained.ckpt' \
    --obj "dog sitting in the library" --desc "" \
    -n "NearFocus_Test" --gpus 0, --logdir ./logs \
    --data_root "./data/near_focus/test.png" \
    --reg_data_root "./data/near_focus/test.png"
    
    
python main.py \
    --base ./v1-finetune_picture.yaml \
    -t --actual_resume './models/pretrained.ckpt' \
    --obj "dog sitting in the library" --desc "near focus camera" \
    -n "NearFocus2_Train" --gpus 0, --logdir ./logs \
    --data_root "./data/near_focus2/train.png" \
    --reg_data_root "./data/near_focus2/train.png"
    
python main.py \
    --base ./v1-finetune_picture.yaml \
    -t --actual_resume './models/pretrained.ckpt' \
    --obj "teddy bear sitting on the bed" --desc "" \
    -n "NearFocus2_Test" --gpus 0, --logdir ./logs \
    --data_root "./data/near_focus2/test.png" \
    --reg_data_root "./data/near_focus2/test.png"
    
python main.py \
    --base ./v1-finetune_picture.yaml \
    -t --actual_resume './models/pretrained.ckpt' \
    --obj "dog sitting in the library" --desc "far focus camera" \
    -n "FarFocus_Train" --gpus 0, --logdir ./logs \
    --data_root "./data/far_focus/train.png" \
    --reg_data_root "./data/far_focus/train.png"
    
python main.py \
    --base ./v1-finetune_picture.yaml \
    -t --actual_resume './models/pretrained.ckpt' \
    --obj "dog sitting in the library" --desc "" \
    -n "FarFocus_Test" --gpus 0, --logdir ./logs \
    --data_root "./data/far_focus/test.png" \
    --reg_data_root "./data/far_focus/test.png"
    
python main.py \
    --base ./v1-finetune_picture.yaml \
    -t --actual_resume './models/pretrained.ckpt' \
    --obj "panda in the forest with some fog" --desc "" \
    -n "Foggy_Train" --gpus 0, --logdir ./logs \
    --data_root "./data/foggy/slight_train.png" \
    --reg_data_root "./data/foggy/slight_train.png"
    
mv ./logs/Foggy_Train/checkpoints/last.ckpt ./logs/Foggy_Train/checkpoints/1.ckpt
    
python main.py \
    --base ./v1-finetune_picture.yaml \
    -t --actual_resume './logs/Foggy_Train/checkpoints/1.ckpt' \
    --obj "panda in the forest with heavy fog" --desc "" \
    -n "Foggy_Train" --gpus 0, --logdir ./logs \
    --data_root "./data/foggy/heavy_train.png" \
    --reg_data_root "./data/foggy/heavy_train.png"
    
    
python main.py \
    --base ./v1-finetune_picture.yaml \
    -t --actual_resume './models/pretrained.ckpt' \
    --obj "panda in the forest" --desc "" \
    -n "Foggy_Test" --gpus 0, --logdir ./logs \
    --data_root "./data/foggy/slight_test.png" \
    --reg_data_root "./data/foggy/slight_test.png"