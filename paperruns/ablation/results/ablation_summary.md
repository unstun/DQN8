# 消融实验推理结果汇总

生成日期: 2026-03-13

变体数: 9, 算法标签数: 10

算法列表: CNN-DDQN, CNN-DDQN+Duel, CNN-DDQN+MD, CNN-DDQN+MHA, CNN-DQN, CNN-DQN+Duel, CNN-DQN+MD, CNN-DQN+MHA, MLP-DDQN, MLP-DQN


## Mode 1: 成功率对比


### Long (≥18m)

| Algorithm     |   Total_Runs |   Successes | SR_pct   |
|:--------------|-------------:|------------:|:---------|
| CNN-DQN+Duel  |           50 |          46 | 92.0%    |
| CNN-DDQN+Duel |           50 |          45 | 90.0%    |
| CNN-DDQN+MD   |           50 |          45 | 90.0%    |
| CNN-DQN+MD    |           50 |          44 | 88.0%    |
| CNN-DQN       |           50 |          40 | 80.0%    |
| CNN-DDQN      |           50 |          39 | 78.0%    |
| CNN-DQN+MHA   |           50 |          36 | 72.0%    |
| MLP-DDQN      |           50 |          18 | 36.0%    |
| MLP-DQN       |           50 |          18 | 36.0%    |
| CNN-DDQN+MHA  |           50 |          10 | 20.0%    |



**排名 (高→低):** CNN-DQN+Duel > CNN-DDQN+Duel > CNN-DDQN+MD > CNN-DQN+MD > CNN-DQN > CNN-DDQN > CNN-DQN+MHA > MLP-DDQN > MLP-DQN > CNN-DDQN+MHA


### Short (6-14m)

| Algorithm     |   Total_Runs |   Successes | SR_pct   |
|:--------------|-------------:|------------:|:---------|
| CNN-DQN+MD    |           50 |          48 | 96.0%    |
| CNN-DDQN+Duel |           50 |          47 | 94.0%    |
| CNN-DDQN+MD   |           50 |          45 | 90.0%    |
| CNN-DQN+MHA   |           50 |          44 | 88.0%    |
| CNN-DDQN      |           50 |          43 | 86.0%    |
| CNN-DQN       |           50 |          43 | 86.0%    |
| CNN-DQN+Duel  |           50 |          43 | 86.0%    |
| MLP-DDQN      |           50 |          33 | 66.0%    |
| MLP-DQN       |           50 |          33 | 66.0%    |
| CNN-DDQN+MHA  |           50 |          29 | 58.0%    |



**排名 (高→低):** CNN-DQN+MD > CNN-DDQN+Duel > CNN-DDQN+MD > CNN-DQN+MHA > CNN-DDQN > CNN-DQN > CNN-DQN+Duel > MLP-DDQN > MLP-DQN > CNN-DDQN+MHA


## Mode 2: 路径质量对比 (全成功 pair)


### Quality Long (≥18m)

全成功 pair 数: **4**

Composite 权重: path_length=1.0, curvature=0.6, plan_time=0.2


| Algorithm     |   N_pairs |   Path_Length_mean |   Path_Length_std |   Curvature_mean |   Curvature_std |   Plan_Time_mean |   Plan_Time_std |   Composite |
|:--------------|----------:|-------------------:|------------------:|-----------------:|----------------:|-----------------:|----------------:|------------:|
| CNN-DQN+MD    |         4 |            24.4744 |            9.5337 |         0.146695 |        0.032077 |          0.32268 |         0.11016 |      0.0436 |
| CNN-DQN+Duel  |         4 |            24.7479 |            9.7913 |         0.170928 |        0.021711 |          0.35452 |         0.10693 |      0.2098 |
| CNN-DQN+MHA   |         4 |            24.8268 |            9.7095 |         0.170913 |        0.027442 |          0.31679 |         0.09912 |      0.2126 |
| CNN-DDQN+Duel |         4 |            25.0151 |            9.4821 |         0.178708 |        0.078343 |          0.25151 |         0.06894 |      0.2582 |
| CNN-DDQN+MD   |         4 |            24.6974 |            9.3193 |         0.191534 |        0.042441 |          0.48481 |         0.15988 |      0.3275 |
| CNN-DQN       |         4 |            25.2861 |            9.6266 |         0.177066 |        0.084351 |          0.33073 |         0.07887 |      0.3489 |
| CNN-DDQN      |         4 |            25.111  |            9.2386 |         0.204639 |        0.069041 |          0.21799 |         0.05726 |      0.3616 |
| CNN-DDQN+MHA  |         4 |            26.1898 |            9.4407 |         0.23784  |        0.065669 |          0.3543  |         0.11424 |      0.7934 |
| MLP-DQN       |         4 |            26.8372 |           11.2086 |         0.219901 |        0.01866  |          0.22707 |         0.13102 |      0.8271 |
| MLP-DDQN      |         4 |            26.8372 |           11.2086 |         0.219901 |        0.01866  |          0.22946 |         0.12972 |      0.8281 |



**Composite 排名 (低→高，越低越好):** CNN-DQN+MD < CNN-DQN+Duel < CNN-DQN+MHA < CNN-DDQN+Duel < CNN-DDQN+MD < CNN-DQN < CNN-DDQN < CNN-DDQN+MHA < MLP-DQN < MLP-DDQN


**路径长度排名 (短→长):** CNN-DQN+MD < CNN-DDQN+MD < CNN-DQN+Duel < CNN-DQN+MHA < CNN-DDQN+Duel < CNN-DDQN < CNN-DQN < CNN-DDQN+MHA < MLP-DQN < MLP-DDQN


**曲率排名 (小→大):** CNN-DQN+MD < CNN-DQN+MHA < CNN-DQN+Duel < CNN-DQN < CNN-DDQN+Duel < CNN-DDQN+MD < CNN-DDQN < MLP-DQN < MLP-DDQN < CNN-DDQN+MHA


### Quality Short (6-14m)

全成功 pair 数: **15**

Composite 权重: path_length=1.0, curvature=0.6, plan_time=0.2


| Algorithm     |   N_pairs |   Path_Length_mean |   Path_Length_std |   Curvature_mean |   Curvature_std |   Plan_Time_mean |   Plan_Time_std |   Composite |
|:--------------|----------:|-------------------:|------------------:|-----------------:|----------------:|-----------------:|----------------:|------------:|
| CNN-DDQN+MD   |        15 |             8.516  |            1.912  |         0.208107 |        0.093253 |          0.14595 |         0.0394  |      0.0756 |
| CNN-DQN+Duel  |        15 |             8.5155 |            1.9452 |         0.22004  |        0.103681 |          0.09887 |         0.02149 |      0.0911 |
| CNN-DDQN+Duel |        15 |             8.5731 |            1.8942 |         0.222374 |        0.088569 |          0.09423 |         0.01984 |      0.157  |
| CNN-DQN+MD    |        15 |             8.5355 |            1.9148 |         0.223608 |        0.08589  |          0.17712 |         0.03623 |      0.2028 |
| CNN-DQN+MHA   |        15 |             8.6104 |            1.8623 |         0.2344   |        0.093264 |          0.12248 |         0.02502 |      0.282  |
| CNN-DDQN      |        15 |             8.627  |            1.8542 |         0.234653 |        0.103539 |          0.1282  |         0.03095 |      0.3055 |
| CNN-DQN       |        15 |             8.7886 |            2.0222 |         0.23099  |        0.123061 |          0.10349 |         0.03435 |      0.4278 |
| CNN-DDQN+MHA  |        15 |             8.8954 |            1.8971 |         0.261689 |        0.092564 |          0.18436 |         0.04783 |      0.7669 |
| MLP-DDQN      |        15 |             9.063  |            1.9272 |         0.274179 |        0.083818 |          0.06595 |         0.02032 |      0.8889 |
| MLP-DQN       |        15 |             9.063  |            1.9272 |         0.274179 |        0.083818 |          0.06917 |         0.02759 |      0.8919 |



**Composite 排名 (低→高，越低越好):** CNN-DDQN+MD < CNN-DQN+Duel < CNN-DDQN+Duel < CNN-DQN+MD < CNN-DQN+MHA < CNN-DDQN < CNN-DQN < CNN-DDQN+MHA < MLP-DDQN < MLP-DQN


**路径长度排名 (短→长):** CNN-DQN+Duel < CNN-DDQN+MD < CNN-DQN+MD < CNN-DDQN+Duel < CNN-DQN+MHA < CNN-DDQN < CNN-DQN < CNN-DDQN+MHA < MLP-DDQN < MLP-DQN


**曲率排名 (小→大):** CNN-DDQN+MD < CNN-DQN+Duel < CNN-DDQN+Duel < CNN-DQN+MD < CNN-DQN < CNN-DQN+MHA < CNN-DDQN < CNN-DDQN+MHA < MLP-DDQN < MLP-DQN
