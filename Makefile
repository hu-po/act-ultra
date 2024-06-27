IDX=0
NUM_EPISODES=10

record_sim:
	poetry run python3 record_sim_episodes.py \
	--task_name sim_pickup_scripted \
	--dataset_dir ../local_data/sim_pickup_scripted \
	--num_episodes ${NUM_EPISODES} 

visualize_sim:
	poetry run python3 visualize_episodes.py --dataset_dir ../local_data/sim_pickup --episode_idx ${IDX} 

train:
	poetry run python3 imitate_episodes.py \
	--task_name sim_pickup_scripted \
	--ckpt_dir ../local_data/sim_pickup_checkpts \
	--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
	--num_epochs 2000  --lr 1e-5 \
	--seed 0

eval:
	poetry run python3 imitate_episodes.py \
	--task_name sim_pickup_scripted \
	--ckpt_dir ../local_data/sim_pickup_checkpts \
	--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
	--num_epochs 2000  --lr 1e-5 \
	--seed 0 \
	--eval

record_sim_cloud:
	export DISPLAY=:1 && $(MAKE) record_sim

visualize_sim_cloud:
	export DISPLAY=:1 && $(MAKE) visualize_sim

train_cloud:
	export DISPLAY=:1 && $(MAKE) train

eval_cloud:
	export DISPLAY=:1 && $(MAKE) eval 
