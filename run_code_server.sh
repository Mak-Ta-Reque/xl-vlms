#srun -p A100-IML --ntasks 1  --gpus-per-task 5 --mem-per-cpu=16G --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh   --container-mounts=/netscratch/software:/netscratch/software:ro,/netscratch/kadir:/netscratch/kadir,"`pwd`":"`pwd`"   --container-workdir="`pwd`"  --time=02:00:00 code_server.sh 

#srun -p A100-IML --ntasks 1  --gpus-per-task 1 --mem-per-cpu=64G --container-image=/netscratch/$USER/enroot/xl_vlm.sqsh   --container-save=/netscratch/$USER/enroot/test.sqsh --container-mounts=/netscratch/software:/netscratch/software:ro,/netscratch/kadir:/netscratch/kadir,/ds:/ds,"`pwd`":"`pwd`",/home/$SLURM_JOB_USER/:/home/$SLURM_JOB_USER/ --container-workdir="`pwd`" --task-prolog="`pwd`/install.sh"  --time=10:00:00 code_server.sh


srun -p A100-IML --ntasks 1  --gpus-per-task 1 --mem-per-cpu=64G --container-image=/netscratch/kadir/enroot/xl_vlm.sqsh --container-mounts=/netscratch/software:/netscratch/software:ro,/netscratch/kadir:/netscratch/kadir,/ds:/ds,"`pwd`":"`pwd`",/home/$SLURM_JOB_USER/:/home/$SLURM_JOB_USER/ --container-workdir="`pwd`" --time=10:00:00 --task-prolog="`pwd`/install.sh" code_server.sh

