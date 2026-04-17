dqn:
	uv run src/scripts/run_dqn.py -cfg experiments/dqn/cartpole.yaml --eval_interval 2500

d-dqn:
	uv run src/scripts/run_dqn.py -cfg experiments/dqn/lunarlander.yaml

d-dqn2:
	nohup caffeinate -dims \
		uv run src/scripts/run_dqn.py -cfg experiments/dqn/mspacman.yaml \
		> d-dqn.log 2>&1 &

sac-pendulum:
	uv run src/scripts/run_sac.py -cfg experiments/sac/sanity_invertedpendulum.yaml

sac-halfcheetah:
	nohup caffeinate -dims \
		uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah.yaml \
		> sac.log 2>&1 &

sac-autotune-pendulum:
	uv run src/scripts/run_sac.py -cfg experiments/sac/sanity_invertedpendulum_autotune.yaml

sac-autotune-halfcheetah:
	nohup caffeinate -dims \
		uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah_autotune.yaml \
		> halfcheetah.log 2>&1 &

sac-clipped-double-Q:
	uv run src/scripts/run_sac.py -cfg experiments/sac/hopper_singleq.yaml
	uv run src/scripts/run_sac.py -cfg experiments/sac/hopper_clipq.yaml