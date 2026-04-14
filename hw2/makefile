.PHONY: reward

reward_bg:
	nohup caffeinate -dimsu $(MAKE) reward > reward.log 2>&1 &

reward:
	uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
		--exp_name cartpole \
		--seed 42

	uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
		-rtg --exp_name cartpole_rtg \
		--seed 42

	uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
		-na --exp_name cartpole_na \
		--seed 42

	uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
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

baseline_bg:
	nohup caffeinate -dimsu $(MAKE) baseline > baseline.log 2>&1 &

baseline:
# No baseline
	uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
		--discount 0.95 -lr 0.01 --exp_name cheetah

# Baseline
	uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
		--discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline

# No baseline na
	uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
		--discount 0.95 -lr 0.01 --exp_name cheetah \
		-na

# Baseline na
	uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
		--discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline \
		-na

LAMBDA ?= 1
LAMBDA_LIST ?= 0 0.95 0.98 0.99 1

gae_sweep_bg:
	nohup caffeinate -dimsu $(MAKE) gae_sweep > gae_sweep.log 2>&1 &

gae_sweep:
	set -euo pipefail; \
	for lambda in $(LAMBDA_LIST); do \
		$(MAKE) gae LAMBDA=$$lambda > output_$$lambda.log 2>& 1; \
	done

gae:
	uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 \
		-n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline \
		--gae_lambda $(LAMBDA) --exp_name lunar_lander_lambda$(LAMBDA)

pendulum:
#	uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 \
		--exp_name pendulum

#	uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 \
		--exp_name pendulum \
		-rtg \
		-na \
		--use_baseline

	uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 2000 -eb 1000 \
		--exp_name pendulum \
		-rtg \
		-na \
		--use_baseline \
		--gae_lambda 0.98 \
		--discount 0.995 \
		-l 3 \
		-s 128 \
		-lr 0.01 \
		-blr 0.01 \
		-bgs 10