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
	uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
		--discount 0.95 -lr 0.01 --exp_name cheetah

# Baseline
#	uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
		--discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline
