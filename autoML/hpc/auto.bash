#!/bin/bash

# 配置
BASH_FILE="/home_cu/u230292/Projects/automl_DOE/hpc/run_LARs.bash"  # 需要提交的 bash 文件路径
PYTHON_FILE="/home_cu/u230292/Projects/automl_DOE/python/run_xxc.py"  # Python 文件路径
REPEAT=5  # 提交 bash 文件的次数
CHECK_INTERVAL=30  # 检查间隔时间（秒）
TASK_LIMIT=1024  # 每组任务的最大数量

# 记录已提交任务的次数
SUBMIT_COUNT=0

while [ $SUBMIT_COUNT -lt $REPEAT ]; do
    # 检查当前任务队列中是否有运行中的任务
    RUNNING_JOBS=$(squeue -u $USER | grep -c " R ")

    if [ "$RUNNING_JOBS" -eq "0" ]; then
        # 动态修改 Python 文件中的 task_id 变量
        OFFSET=$(($SUBMIT_COUNT * $TASK_LIMIT))
        sed -i "s/'task_id': int(os.getenv('SLURM_ARRAY_TASK_ID'))/'task_id': int(os.getenv('SLURM_ARRAY_TASK_ID')) + $OFFSET/g" "$PYTHON_FILE"
        
        # 提交 bash 文件
        bash "$BASH_FILE"
        
        # 更新已提交任务的次数
        SUBMIT_COUNT=$(($SUBMIT_COUNT + 1))
        
        echo "Submitted job number $SUBMIT_COUNT with offset $OFFSET."
        
        # 延迟 5 秒
        sleep 5
        
        # 等待任务完成
        while true; do
            RUNNING_JOBS=$(squeue -u $USER | grep -c " R ")
            echo "Current running jobs: $RUNNING_JOBS"
            if [ "$RUNNING_JOBS" -eq "0" ]; then
                break
            fi
            sleep $CHECK_INTERVAL
        done
        
        # 恢复 Python 文件中的原始 task_id 变量
        sed -i "s/'task_id': int(os.getenv('SLURM_ARRAY_TASK_ID')) + $OFFSET/'task_id': int(os.getenv('SLURM_ARRAY_TASK_ID'))/g" "$PYTHON_FILE"
    else
        echo "Jobs are still running. Waiting for $CHECK_INTERVAL seconds..."
        sleep $CHECK_INTERVAL
    fi
done

echo "All $REPEAT jobs have been submitted."
