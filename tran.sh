python -m torch.distributed.run  \
  --nproc_per_node=2 \
  --nnodes=1 \
  --master_port 29500 \
  main.py \
  --cfg=configs/transnet/transnet_tiny_64.yaml \
  --batch_size=64 \
  --epoch=300 \
  --tag=ADNet \
  --gpu=0,1
