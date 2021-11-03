sbatch -J ifg_gradients \
       -c 12 \
       -p IB_44C_512G \
       -q pq_nbc \
       --account iacc_nbc \
       --wrap="python3 rsfc_workflow.py"
