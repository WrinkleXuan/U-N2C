# U-N2C

This repository contains the code  of our paper: U-N2C: Dual Memory-guided Disentanglement Network for Unsupervised System Matrix Denoising in Magnetic Particle Imaging.

The paper is currently under review in IEEE Transactions on Image Processing.

## The framework of our U-N2C
![framework](https://github.com/user-attachments/assets/67a694da-9954-4c51-9565-0497a8eb41ee)

## Models
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-7btt">Gaussion</th>
    <th class="tg-7btt">Heteroscedastic Gaussian</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7btt">SNR=0.5</td>
    <td class="tg-0pky"><a href="https://drive.google.com/file/d/1e9FHpdcNtRImpHdTi00MfRpxB2Ine6lV/view?usp=drive_link">cket_best</a></td>
    <td class="tg-0pky"><a href="https://drive.google.com/file/d/1sxYWKas_wlcH84lCFnUdUhs6IaHRAiYW/view?usp=drive_link">ckpt_best</a></td>
  </tr>
</tbody>
</table>


## Training
    sh train.sh

## Inference
    sh test.sh 


## Requirments

* Python 3.10.0
* Pytorch 2.0.0
