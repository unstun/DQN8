# diag 消融实验推理结果汇总

edt_collision_margin = **diag** (√2/2 补偿)

变体数: 9, 算法标签数: 10

算法列表: CNN-DDQN, CNN-DDQN+Duel, CNN-DDQN+MD, CNN-DDQN+MHA, CNN-DQN, CNN-DQN+Duel, CNN-DQN+MD, CNN-DQN+MHA, MLP-DDQN, MLP-DQN


## Mode 1: 成功率对比


### Long (≥18m)

| Algorithm     |   Total_Runs |   Successes | SR_pct   |
|:--------------|-------------:|------------:|:---------|
| CNN-DDQN+Duel |           50 |          46 | 92.0%    |
| CNN-DQN       |           50 |          45 | 90.0%    |
| CNN-DDQN+MD   |           50 |          44 | 88.0%    |
| MLP-DQN       |           50 |          43 | 86.0%    |
| CNN-DQN+MD    |           50 |          42 | 84.0%    |
| CNN-DQN+MHA   |           50 |          41 | 82.0%    |
| MLP-DDQN      |           50 |          41 | 82.0%    |
| CNN-DDQN+MHA  |           50 |          41 | 82.0%    |
| CNN-DDQN      |           50 |          40 | 80.0%    |
| CNN-DQN+Duel  |           50 |          40 | 80.0%    |



**排名 (高→低):** CNN-DDQN+Duel > CNN-DQN > CNN-DDQN+MD > MLP-DQN > CNN-DQN+MD > CNN-DQN+MHA > MLP-DDQN > CNN-DDQN+MHA > CNN-DDQN > CNN-DQN+Duel


### Short (6-14m)

| Algorithm     |   Total_Runs |   Successes | SR_pct   |
|:--------------|-------------:|------------:|:---------|
| CNN-DDQN+MD   |           50 |          46 | 92.0%    |
| CNN-DQN+Duel  |           50 |          46 | 92.0%    |
| CNN-DQN+MD    |           50 |          45 | 90.0%    |
| CNN-DQN+MHA   |           50 |          44 | 88.0%    |
| CNN-DDQN      |           50 |          43 | 86.0%    |
| CNN-DQN       |           50 |          43 | 86.0%    |
| MLP-DQN       |           50 |          43 | 86.0%    |
| CNN-DDQN+Duel |           50 |          43 | 86.0%    |
| CNN-DDQN+MHA  |           50 |          42 | 84.0%    |
| MLP-DDQN      |           50 |          41 | 82.0%    |



**排名 (高→低):** CNN-DDQN+MD > CNN-DQN+Duel > CNN-DQN+MD > CNN-DQN+MHA > CNN-DDQN > CNN-DQN > MLP-DQN > CNN-DDQN+Duel > CNN-DDQN+MHA > MLP-DDQN


## Mode 2: 路径质量对比 (全成功 pair)


### Quality Long (≥18m)

全成功 pair 数: **25**

Composite 权重: path_length=1.0, curvature=0.6, plan_time=0.2


| Algorithm     |   N_pairs |   Path_Length_mean |   Path_Length_std |   Curvature_mean |   Curvature_std |   Plan_Time_mean |   Plan_Time_std |   Composite |
|:--------------|----------:|-------------------:|------------------:|-----------------:|----------------:|-----------------:|----------------:|------------:|
| CNN-DDQN+Duel |        25 |            26.005  |            6.1284 |         0.148373 |        0.0492   |          0.43703 |         0.11697 |      0.1097 |
| CNN-DDQN+MD   |        25 |            26.0683 |            6.3108 |         0.147457 |        0.046849 |          0.34767 |         0.07745 |      0.1211 |
| CNN-DQN+MD    |        25 |            26.1425 |            6.3395 |         0.15418  |        0.041566 |          0.36237 |         0.08268 |      0.2675 |
| CNN-DQN       |        25 |            26.1286 |            6.3055 |         0.159903 |        0.041838 |          0.32149 |         0.09374 |      0.3103 |
| CNN-DQN+Duel  |        25 |            26.2941 |            6.1089 |         0.157444 |        0.04003  |          0.33991 |         0.07585 |      0.4243 |
| MLP-DQN       |        25 |            26.2514 |            6.1573 |         0.166083 |        0.060793 |          0.14543 |         0.03744 |      0.4261 |
| CNN-DDQN      |        25 |            26.2839 |            6.2546 |         0.160114 |        0.06104  |          0.30398 |         0.07426 |      0.4355 |
| MLP-DDQN      |        25 |            26.3882 |            6.3084 |         0.169467 |        0.045537 |          0.14419 |         0.03429 |      0.5792 |
| CNN-DDQN+MHA  |        25 |            26.2779 |            6.1769 |         0.173067 |        0.059937 |          0.47348 |         0.10524 |      0.6419 |
| CNN-DQN+MHA   |        25 |            26.6759 |            6.2384 |         0.175468 |        0.046775 |          0.42143 |         0.13306 |      0.9824 |



**Composite 排名 (低→高，越低越好):** CNN-DDQN+Duel < CNN-DDQN+MD < CNN-DQN+MD < CNN-DQN < CNN-DQN+Duel < MLP-DQN < CNN-DDQN < MLP-DDQN < CNN-DDQN+MHA < CNN-DQN+MHA


**路径长度排名 (短→长):** CNN-DDQN+Duel < CNN-DDQN+MD < CNN-DQN < CNN-DQN+MD < MLP-DQN < CNN-DDQN+MHA < CNN-DDQN < CNN-DQN+Duel < MLP-DDQN < CNN-DQN+MHA


**曲率排名 (小→大):** CNN-DDQN+MD < CNN-DDQN+Duel < CNN-DQN+MD < CNN-DQN+Duel < CNN-DQN < CNN-DDQN < MLP-DQN < MLP-DDQN < CNN-DDQN+MHA < CNN-DQN+MHA


### Quality Short (6-14m)

全成功 pair 数: **18**

Composite 权重: path_length=1.0, curvature=0.6, plan_time=0.2


| Algorithm     |   N_pairs |   Path_Length_mean |   Path_Length_std |   Curvature_mean |   Curvature_std |   Plan_Time_mean |   Plan_Time_std |   Composite |
|:--------------|----------:|-------------------:|------------------:|-----------------:|----------------:|-----------------:|----------------:|------------:|
| CNN-DDQN+MD   |        18 |             8.1221 |            2.1849 |         0.203575 |        0.097579 |          0.12187 |         0.0293  |      0.1512 |
| MLP-DDQN      |        18 |             8.1814 |            2.2419 |         0.184864 |        0.123773 |          0.05384 |         0.01407 |      0.1709 |
| CNN-DQN+MD    |        18 |             8.1275 |            2.1682 |         0.203679 |        0.122885 |          0.1297  |         0.03179 |      0.1763 |
| CNN-DDQN+Duel |        18 |             8.1358 |            2.1241 |         0.210033 |        0.102235 |          0.12234 |         0.03515 |      0.2158 |
| CNN-DDQN      |        18 |             8.1579 |            2.1591 |         0.200827 |        0.104342 |          0.14834 |         0.04898 |      0.275  |
| CNN-DQN       |        18 |             8.2576 |            2.1487 |         0.228506 |        0.066617 |          0.12718 |         0.03643 |      0.6427 |
| MLP-DQN       |        18 |             8.2305 |            2.2633 |         0.27247  |        0.324086 |          0.08913 |         0.14408 |      0.6872 |
| CNN-DQN+Duel  |        18 |             8.2967 |            2.0488 |         0.234846 |        0.116712 |          0.14003 |         0.04102 |      0.7946 |
| CNN-DQN+MHA   |        18 |             8.312  |            2.2498 |         0.240489 |        0.101359 |          0.13124 |         0.03027 |      0.8499 |
| CNN-DDQN+MHA  |        18 |             8.3149 |            2.1285 |         0.250377 |        0.141626 |          0.13251 |         0.03685 |      0.8973 |



**Composite 排名 (低→高，越低越好):** CNN-DDQN+MD < MLP-DDQN < CNN-DQN+MD < CNN-DDQN+Duel < CNN-DDQN < CNN-DQN < MLP-DQN < CNN-DQN+Duel < CNN-DQN+MHA < CNN-DDQN+MHA


**路径长度排名 (短→长):** CNN-DDQN+MD < CNN-DQN+MD < CNN-DDQN+Duel < CNN-DDQN < MLP-DDQN < MLP-DQN < CNN-DQN < CNN-DQN+Duel < CNN-DQN+MHA < CNN-DDQN+MHA


**曲率排名 (小→大):** MLP-DDQN < CNN-DDQN < CNN-DDQN+MD < CNN-DQN+MD < CNN-DDQN+Duel < CNN-DQN < CNN-DQN+Duel < CNN-DQN+MHA < CNN-DDQN+MHA < MLP-DQN
