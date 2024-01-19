model_name=ReactionMGMTurnNet
dataset_name=uspto
cross_attn_h_rate=0.5
gpu=0


python -u rxn_branch_pretraining.py \
--gpu ${gpu} \
--model_name ${model_name} \
--dataset_name ${dataset_name} \
--batch_size 64 \
--learning_rate 2e-4 \
--num_workers 6 \
--node_out_feats 64 \
--attention_heads 8 \
--attention_layers 8 \
--perform_mask \
--use_atom_envs_type \
--evaluate_in_train \
--evaluate_pre_step 5000 \
--cross_attn_h_rate ${cross_attn_h_rate} > logs/pretrain_rxn_attn_net_in_${dataset_name}_up_head_layer_${model_name}_cross_attn_h_rate_${cross_attn_h_rate}.log &