sac-bc-antmaze:
	uv run src/scripts/run.py \
	--njobs=4 \
	"JOB --run_group=q1 --base_config=sacbc --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=42 --alpha=1" \
	"JOB --run_group=q1 --base_config=sacbc --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=42 --alpha=3" \
	"JOB --run_group=q1 --base_config=sacbc --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=42 --alpha=10" \
	"JOB --run_group=q1 --base_config=sacbc --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=42 --alpha=30"

sac-bc-cube:
	uv run src/scripts/run.py --njobs=4 \
	"JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=42 --alpha=30" \
	"JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=42 --alpha=100" \
	"JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=42 --alpha=300" \
	"JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=42 --alpha=1000" \

sac-bc-arena:
	uv run src/scripts/run.py --njobs=4 \
	"JOB --run_group=q1 --base_config=sacbc --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=42 --alpha=1" \
	"JOB --run_group=q1 --base_config=sacbc --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=42 --alpha=3" \
	"JOB --run_group=q1 --base_config=sacbc --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=42 --alpha=10" \
	"JOB --run_group=q1 --base_config=sacbc --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=42 --alpha=30" \