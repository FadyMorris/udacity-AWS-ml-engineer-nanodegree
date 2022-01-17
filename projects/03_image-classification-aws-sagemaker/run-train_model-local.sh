export SM_CHANNEL_TRAINING=./data/dogImages/
export SM_MODEL_DIR=./debug/model/
export SM_OUTPUT_DATA_DIR=./debug/output/
python3 ./code/train_model.py --data ./data/dogImages/ --learning_rate 0.02 --batch_size 32 --model_dir ./debug/model/ --output_dir ./debug/output/
