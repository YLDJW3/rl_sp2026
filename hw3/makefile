dqn:
	uv run src/scripts/run_dqn.py -cfg experiments/dqn/cartpole.yaml --eval_interval 2500

d-dqn:
	uv run src/scripts/run_dqn.py -cfg experiments/dqn/lunarlander.yaml

d-dqn2:
	nohup caffeinate -dims \
		uv run src/scripts/run_dqn.py -cfg experiments/dqn/mspacman.yaml \
		> d-dqn.log 2>&1 &
