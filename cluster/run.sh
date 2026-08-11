/usr/local/openmpi/bin/mpirun \
    -hostfile hostfile.16.run.16 \
    -np 16 \
    --allow-run-as-root \
    --timeout 360 \
    --verbose \
    -mca plm_base_verbose 5 \
    -mca orte_abort_timeout 300 \
    -mca btl_tcp_if_include bond0 \
    --prtemca prte_keep_fqdn_hostnames 1 \
    --bind-to none \
    -x LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/usr/lib/:/usr/lib64:/usr/local/musa/lib:/usr/local/musa/mudnn/lib:/usr/local/openmpi/lib \
    -x PATH=/usr/local/musa/bin:/usr/local/musa/mudnn/bin:/usr/local/musa/mudnn/mudnn_bench/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    -x MCCL_PROTOS=2 \
    -x MCCL_ALGOS=1 \
    -x MCCL_CROSS_NIC=1 \
    -x MCCL_IB_TIMEOUT=20 \
    -x MCCL_IB_RETRY_CNT=7 \
    /usr/local/musa/mccl_test/all_reduce_perf -b 512K -e 16G -f 2 -g 1 
