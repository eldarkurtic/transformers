#!/bin/bash

GPU=0

ROOT=/hdd/src/neuralmagic/transformers/examples/pytorch/question-answering

RECIPE_DIR=$ROOT/recipes/mfac
RECIPE_NAME=uni_prune90_lr_rewind

MODEL_DIR=$ROOT/models/mfac
MODEL_NAME=$RECIPE_NAME

python $ROOT/run_qa.py \
  --model_name_or_path bert-base-uncased \
  --max_train_samples 64 \
  --max_eval_samples 64 \
  --distill_teacher $ROOT/models/release_0.5/bert-base-12layers \
  --dataset_name squad \
  --do_train \
  --fp16 \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 2 \
  --logging_steps 2 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $MODEL_DIR/$MODEL_NAME \
  --overwrite_output_dir \
  --cache_dir cache \
  --preprocessing_num_workers 16 \
  --seed 42 \
  --num_train_epochs 30 \
  --save_steps 1000 \
  --save_total_limit 2 \
  --recipe $RECIPE_DIR/$RECIPE_NAME.md \
  --report_to wandb


#  --learning_rate 1e-4 \
