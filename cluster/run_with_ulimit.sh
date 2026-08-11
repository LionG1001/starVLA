#!/bin/bash
# 在 MPI 程序启动前设置 ulimit
ulimit -l unlimited
ulimit -s unlimited
ulimit -n 1024000
# 执行你的 MPI 程序，并传递所有参数
exec "$@"
