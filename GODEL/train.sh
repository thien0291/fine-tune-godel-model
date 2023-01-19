# DATA_NAME="../data/favs_data/"
TRAIN_FILE="../data/favs_data/sample.json"
VALIDATION_FILE="../data/favs_data/sample.json"
TEST_FILE="../data/favs_data/sample.json"
OUTPUT_DIR="../trained_models/"
MODEL_PATH="../GODEL-Base"
EXP_NAME="exp_name"

# --dataset_name ${DATA_NAME} \

python train.py --model_name_or_path ${MODEL_PATH} \
	--train_file ${TRAIN_FILE} \
	--validation_file ${VALIDATION_FILE} \
	--test_file ${TEST_FILE} \
	--output_dir ${OUTPUT_DIR} \
	--per_device_train_batch_size=16 \
	--per_device_eval_batch_size=16 \
	--max_target_length 512 \
	--max_length 512 \
	--num_train_epochs 50 \
	--save_steps 10000 \
	--num_beams 5 \
	--exp_name ${EXP_NAME} --preprocessing_num_workers 24