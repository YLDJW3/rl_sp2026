sac-bc-antmaze:
	mkdir -p logs && nohup uv run src/scripts/run.py \
	--njobs=4 \
	"JOB --run_group=q1 --base_config=sacbc --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=42 --alpha=1" \
	"JOB --run_group=q1 --base_config=sacbc --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=42 --alpha=3" \
	"JOB --run_group=q1 --base_config=sacbc --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=42 --alpha=10" \
	"JOB --run_group=q1 --base_config=sacbc --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=42 --alpha=30" \
	> logs/sac-bc-antmaze.log 2>&1 &

sac-bc-cube:
	mkdir -p logs && nohup uv run src/scripts/run.py --njobs=4 \
	"JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=42 --alpha=30" \
	"JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=42 --alpha=100" \
	"JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=42 --alpha=300" \
	"JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=42 --alpha=1000" \
	> logs/sac-bc-cube.log 2>&1 &

sac-bc-arena:
	mkdir -p logs && nohup uv run src/scripts/run.py --njobs=4 \
	"JOB --run_group=q1 --base_config=sacbc --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=42 --alpha=1" \
	"JOB --run_group=q1 --base_config=sacbc --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=42 --alpha=3" \
	"JOB --run_group=q1 --base_config=sacbc --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=42 --alpha=10" \
	"JOB --run_group=q1 --base_config=sacbc --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=42 --alpha=30" \
	> logs/sac-bc-arena.log 2>&1 &

iql-debug:
	mkdir -p logs && nohup uv run src/scripts/run.py --run_group=q2 --base_config=iql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=42 --alpha=10 \
	> logs/iql-debug.log 2>&1 &

iql-cube:
	mkdir -p logs && nohup uv run src/scripts/run.py --njobs=3 \
	"JOB --run_group=q2 --base_config=iql --env_name=cube-single-play-singletask-task1-v0 --seed=42 --alpha=1" \
	"JOB --run_group=q2 --base_config=iql --env_name=cube-single-play-singletask-task1-v0 --seed=42 --alpha=3" \
	"JOB --run_group=q2 --base_config=iql --env_name=cube-single-play-singletask-task1-v0 --seed=42 --alpha=10" \
	> logs/iql-cube.log 2>&1 &

iql-arena:
	mkdir -p logs && nohup uv run src/scripts/run.py --njobs=3 \
	"JOB --run_group=q2 --base_config=iql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=42 --alpha=1" \
	"JOB --run_group=q2 --base_config=iql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=42 --alpha=3" \
	"JOB --run_group=q2 --base_config=iql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=42 --alpha=10" \
	> logs/iql-arena.log 2>&1 &

fql-debug:
	mkdir -p logs && nohup uv run src/scripts/run.py --run_group=q3 --base_config=fql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=42 --alpha=10 \
	> logs/fql-debug.log 2>&1 &

fql-cube:
	mkdir -p logs && nohup uv run src/scripts/run.py --njobs=4 \
	"JOB --run_group=q3 --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=42 --alpha=30" \
	"JOB --run_group=q3 --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=42 --alpha=100" \
	"JOB --run_group=q3 --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=42 --alpha=300" \
	"JOB --run_group=q3 --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=42 --alpha=1000" \
	> logs/fql-cube.log 2>&1 &

fql-arena:
	mkdir -p logs && nohup uv run src/scripts/run.py --njobs=4 \
	"JOB --run_group=q3 --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=42 --alpha=1" \
	"JOB --run_group=q3 --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=42 --alpha=3" \
	"JOB --run_group=q3 --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=42 --alpha=10" \
	"JOB --run_group=q3 --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=42 --alpha=30" \
	> logs/fql-arena.log 2>&1 &
