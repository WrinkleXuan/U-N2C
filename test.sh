python -m torch.distributed.run  \
  --nproc_per_node=1 \
  --nnodes=1 \
  --master_port 29501 \
  main.py \
  --cfg=output/convnet_64_2_4/gaussion_noise_0.5/config.yaml \
  --batch_size=512 \
  --resume=output/convnet_64_2_4/gaussion_noise_0.5/ckpt_best.pth \
  --eval \
  --throughput \
  --gpu=2 \
