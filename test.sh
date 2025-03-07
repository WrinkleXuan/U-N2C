python -m torch.distributed.run  \
  --nproc_per_node=1 \
  --nnodes=1 \
  --master_port 29501 \
  main.py \
  --cfg=output/transnet_tiny_64/ADNet2025-02-19-03-15-18/config.yaml \
  --batch_size=512 \
  --resume=output/transnet_tiny_64/ADNet2025-02-19-03-15-18/ckpt_best.pth \
  --tag=ADNet \
  --eval \
  --throughput \
  --gpu=2 \