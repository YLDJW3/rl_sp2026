.PHONY: reward

reward:
#	uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
		--exp_name cartpole \
		--seed 42

#	uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
		-rtg --exp_name cartpole_rtg \
		--seed 42

#	uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
		-na --exp_name cartpole_na \
		--seed 42

#	uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
		-rtg -na --exp_name cartpole_rtg_na \
		--seed 42

	uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
		--exp_name cartpole_lb \
		--seed 42

	uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
		-rtg --exp_name cartpole_lb_rtg \
		--seed 42

	uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
		-na --exp_name cartpole_lb_na \
		--seed 42

	uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
		-rtg -na --exp_name cartpole_lb_rtg_na \
		--seed 42

baseline:
# No baseline
#	uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
		--discount 0.95 -lr 0.01 --exp_name cheetah

# Baseline
#	uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
		--discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline

# Baseline lr=0.02 n=200
#	uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 200 -b 5000 -eb 3000 -rtg \
		--discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline \
		--seed 42

# No baseline na
#	uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
		--discount 0.95 -lr 0.01 --exp_name cheetah \
		-na

# Baseline na
	uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
		--discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline \
		-na --gae_lambda 1

LAMBDA ?= 1
LAMBDA_LIST ?= 0.95 0.98 0.99

gae_sweep:
	for lambda in $(LAMBDA_LIST); do \
		nohup $(MAKE) gae LAMBDA=$$lambda > output_$$lambda.log 2>&1 & \
	done; \
	wait

gae:
	uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 \
		-n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline \
		--gae_lambda $(LAMBDA) --exp_name lunar_lander_lambda$(LAMBDA)

pendulum:
	uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 \
		--exp_name pendulum \
		--discount 0.95 \
		-l 3 -s 128 \
		-lr 0.01 \
		-rtg \
		-na \
		--use_baseline -blr 0.01 -bgs 5 \