# 进度说明

## 课设 PDF（2）（3）完成情况说明


```text
（2）对应：以 MOVING-DRONE / SDNet 为 baseline，在 WuhanMetroCrowd 数据集上完成训练、测试和评价。
（3）对应：针对物理约束和时空一致性问题，对模型进行改进、微调和增强，并通过 MAE、MSE、WRAE 说明改进效果。
```

本课设使用的数据集划分如下：

```text
训练集：场景 1-8
验证集：场景 9-10
测试集：场景 11-12
```

评价指标以课设要求中的 `MAE`、`MSE`、`WRAE` 为主。需要说明的是，当前 SDNet 代码中输出名为 `MSE`，但实际计算方式是均方误差开根号，报告中写为 `MSE/RMSE`。

---

## 一、对应课设 PDF（2）：Baseline 训练与测试

### 1. Baseline 方法

第（2）步采用文献 MOVING-DRONE 中的 **SDNet** 方法作为 baseline，代码参考：

```text
D:\模式识别与机器学习课设\MovingDroneCrowd-main
```

使用的数据集为已转换后的 WuhanMetroCrowd：

```text
D:\模式识别与机器学习课设\MovingDroneCrowd-main\data\WuhanMetroCrowd_SDNet
```

训练初始化使用官方 SDNet pretrained counter 权重：

```text
D:\模式识别与机器学习课设\MovingDroneCrowd-main\pre_train_model\SDNet_pre_trained_counter_MDC_VGG16_FPN_ep_200_downscale_16.pth
```

### 2. 30 epoch 训练记录

老师要求 baseline 训练 30 epoch，目前已经完成。

训练命令为：

```bash
python scripts/run_wuhan_sdnet_quick.py --name wuhan_sdnet_counter_e30_640 --epochs 30 --train-h 256 --train-w 320 --train-max-long 640 --train-max-short 360 --test-max-long 640 --test-max-short 360 --train-num-workers 0 --val-num-workers 0 --print-freq 500 --val-interval 5 --start-val 5 --skip-flag --pre-train-counter pre_train_model/SDNet_pre_trained_counter_MDC_VGG16_FPN_ep_200_downscale_16.pth --no-test
```

训练输出目录为：

```text
D:\模式识别与机器学习课设\MovingDroneCrowd-main\exp\WuhanMetroCrowd\wuhan_sdnet_counter_e30_640
```

最终训练状态文件为：

```text
D:\模式识别与机器学习课设\MovingDroneCrowd-main\exp\WuhanMetroCrowd\wuhan_sdnet_counter_e30_640\latest_state.pth
```

其中记录的最终状态为：

```text
completed_epoch: 30
i_tb: 31740
```

训练过程中每 5 epoch 在验证集上保存一次 checkpoint，记录如下：

| Epoch | Iter | 验证 MAE | 验证 MSE/RMSE | 验证 seq_MAE | 验证 WRAE | MIAE | MOAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 5290 | 7.819 | 9.579 | 152.951 | 110.712 | 5.346 | 6.317 |
| 10 | 10580 | 7.451 | 8.848 | 16.038 | 11.374 | 2.031 | 1.854 |
| 15 | 15870 | 8.418 | 9.298 | 391.218 | 285.441 | 12.792 | 13.976 |
| 20 | 21160 | 7.083 | 8.937 | 12.453 | 8.953 | 2.242 | 2.124 |
| 25 | 26450 | 7.510 | 8.570 | 65.231 | 45.628 | 3.476 | 3.979 |
| 30 | 31740 | 8.273 | 9.277 | 400.136 | 286.933 | 13.144 | 14.216 |

训练代码按照验证集表现自动选择最优模型，本次 `best_model.pth` 对应第 20 epoch：

```text
D:\模式识别与机器学习课设\MovingDroneCrowd-main\exp\WuhanMetroCrowd\wuhan_sdnet_counter_e30_640\best_model.pth
```

对应 checkpoint 为：

```text
ep_20_iter_21160_mae_7.083_mse_8.937_seq_MAE_12.453_WRAE_8.953_MIAE_2.242_MOAE_2.124.pth
```

### 3. Baseline 测试记录

测试命令为：

```bash
python test.py --MODEL SDNet --DATASET WuhanMetroCrowd --model_path exp/WuhanMetroCrowd/wuhan_sdnet_counter_e30_640/best_model.pth --test_interval 4 --test_split test --skip_flag false --GPU_ID 0 --output_dir test_results --test_name wuhan_sdnet_counter_e30_640 --test_max_long 640 --test_max_short 360
```

测试结果文件为：

```text
D:\模式识别与机器学习课设\MovingDroneCrowd-main\test_results\WuhanMetroCrowd\SDNet_wuhan_sdnet_counter_e30_640_test_4_best_model\results.txt
```

在测试集场景 `11` 和 `12` 上得到 baseline 指标：

| 方法 | MAE | MSE/RMSE | WRAE |
|---|---:|---:|---:|
| SDNet baseline | 28.12 | 28.14 | 171.17 |

这说明第（2）步已经完成：已在 WuhanMetroCrowd 上完成 SDNet baseline 的 30 epoch 训练、最优模型选择和测试集评价。

---

## 二、对应课设 PDF（3）：物理约束与时空一致性改进

### 1. 改进目标

第（3）步要求在 WuhanMetroCrowd 数据集上，针对物理约束和时空一致性等问题完成模型改进与微调。行人流量计数任务中，相邻帧之间应满足基本人数守恒关系：

```text
N_next = N_prev - Out + In
```

其中：

```text
N_prev：前一帧人数
N_next：后一帧人数
Out：从前一帧到后一帧离开的行人数
In：从前一帧到后一帧进入的行人数
```

因此，第三步围绕“相邻帧人数变化”和“流入/流出一致性”进行改进。

### 2. 训练阶段微调方式

首先在 SDNet 训练代码中加入了物理约束和时空一致性损失项，核心修改文件为：

```text
D:\模式识别与机器学习课设\MovingDroneCrowd-main\model\VIC.py
D:\模式识别与机器学习课设\MovingDroneCrowd-main\scripts\run_wuhan_sdnet_quick.py
```

新增损失项包括：

| 损失项 | 作用 |
|---|---|
| `PHYSICAL_LOSS_WEIGHT` | 约束 `N_next` 接近 `N_prev - Out + In` |
| `TEMPORAL_COUNT_LOSS_WEIGHT` | 监督相邻帧人数变化 |
| `FLOW_BALANCE_LOSS_WEIGHT` | 监督流入/流出变化与人数变化一致 |
| `SHARE_CONSISTENCY_LOSS_WEIGHT` | 约束相邻帧共享行人数量一致性 |

### 3. 第三步微调 epoch 记录

第三步从第二步 30 epoch baseline 的最优模型出发继续微调：

```text
exp\WuhanMetroCrowd\wuhan_sdnet_counter_e30_640\best_model.pth
```

#### 3.1 物理约束微调 5 epoch

实验名称：

```text
wuhan_sdnet_physical_e5_640_from_e30_best
```

输出目录：

```text
D:\模式识别与机器学习课设\MovingDroneCrowd-main\exp\WuhanMetroCrowd\wuhan_sdnet_physical_e5_640_from_e30_best
```

使用的主要约束权重为：

```text
physical-loss-weight = 0.05
share-consistency-loss-weight = 0.02
```

5 epoch 验证记录如下：

| Epoch | Iter | 验证 MAE | 验证 MSE/RMSE | 验证 seq_MAE | 验证 WRAE | MIAE | MOAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1058 | 23.654 | 28.227 | 8.185 | 5.847 | 2.175 | 2.409 |
| 2 | 2116 | 17.065 | 19.993 | 66.073 | 47.227 | 2.845 | 1.889 |
| 3 | 3174 | 13.305 | 16.113 | 427.979 | 311.322 | 14.187 | 15.330 |
| 4 | 4232 | 13.582 | 16.272 | 112.874 | 82.893 | 4.208 | 5.088 |
| 5 | 5290 | 12.704 | 15.673 | 16.434 | 11.382 | 1.983 | 1.805 |

该 5 epoch 微调在验证集部分指标上有波动，但在测试集上没有超过第二步 baseline。其 `best_model.pth` 测试结果为：

| 方法 | MAE | MSE/RMSE | WRAE |
|---|---:|---:|---:|
| physical loss fine-tuning | 81.41 | 82.93 | 451.76 |

因此这组训练型微调可作为第三步的“物理约束损失尝试/消融记录”，但不作为最终改进结果。

#### 3.2 时序监督微调 1 epoch

实验名称：

```text
wuhan_sdnet_temporal_e1_640_from_e30_best
```

输出目录：

```text
D:\模式识别与机器学习课设\MovingDroneCrowd-main\exp\WuhanMetroCrowd\wuhan_sdnet_temporal_e1_640_from_e30_best
```

使用的主要约束权重为：

```text
temporal-count-loss-weight = 0.02
flow-balance-loss-weight = 0.02
share-consistency-loss-weight = 0.005
```

1 epoch 验证记录如下：

| Epoch | Iter | 验证 MAE | 验证 MSE/RMSE | 验证 seq_MAE | 验证 WRAE | MIAE | MOAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1058 | 10.082 | 11.392 | 296.242 | 209.028 | 9.499 | 10.628 |

该实验说明较大的时序监督权重会导致视频级累计指标不稳定，因此未作为最终方案。

### 4. 最终采用的改进方案：推理阶段时空物理一致性增强

由于直接在训练阶段加入较强约束会造成模型退化，最终第三步采用更稳定的方式：保留第二步训练得到的 SDNet baseline 权重，在推理阶段对相邻帧的流入/流出预测进行物理一致性修正。

新增脚本为：

```text
D:\模式识别与机器学习课设\MovingDroneCrowd-main\scripts\evaluate_wuhan_temporal_refine.py
```

该脚本的核心思想是：模型先正常输出每一对相邻帧的 `N_prev`、`N_next`、`In`、`Out`，然后将流入/流出计数修正到更符合：

```text
N_next = N_prev - Out + In
```

修正强度 `alpha` 不使用测试集调参，而是在验证集场景 `9、10` 上选择，再应用到测试集场景 `11、12`。本次验证集选择结果为：

```text
alpha = 1.00
```

运行命令为：

```bash
python scripts/evaluate_wuhan_temporal_refine.py --model-path exp/WuhanMetroCrowd/wuhan_sdnet_counter_e30_640/best_model.pth --gpu 0 --test-interval 4 --skip-flag false --test-max-long 640 --test-max-short 360 --name wuhan_sdnet_temporal_refine_e30_640 --alpha-grid 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
```

输出目录为：

```text
D:\模式识别与机器学习课设\MovingDroneCrowd-main\test_results\WuhanMetroCrowd\SDNet_wuhan_sdnet_temporal_refine_e30_640_val2test_4_best_model
```

其中主要结果文件包括：

```text
results.txt
metrics.csv
scene_counts.csv
alpha_search_val.csv
metric_comparison.png
scene_count_comparison.png
summary.json
```

### 5. 从 MAE、MSE、WRAE 说明改进后优越性

为了公平比较，第三步的 baseline 与 refined 结果使用同一评估脚本重新计算。测试集结果如下：

| 方法 | MAE | MSE/RMSE | WRAE |
|---|---:|---:|---:|
| SDNet baseline | 28.15 | 28.17 | 171.34 |
| Temporal physical refinement | 26.43 | 26.44 | 162.21 |

三个核心指标均下降，说明改进后结果更优：

| 指标 | Baseline | 改进后 | 下降量 | 相对下降 |
|---|---:|---:|---:|---:|
| MAE | 28.15 | 26.43 | 1.72 | 约 6.1% |
| MSE/RMSE | 28.17 | 26.44 | 1.73 | 约 6.1% |
| WRAE | 171.34 | 162.21 | 9.13 | 约 5.3% |

从三个指标含义看：

```text
MAE 下降：说明测试视频场景的平均人数绝对误差降低。
MSE/RMSE 下降：说明较大误差受到惩罚后，整体误差仍然降低。
WRAE 下降：说明按视频长度加权后的相对误差降低，改进方法对视频级行人流量累计计数更稳定。
```

因此，第三步最终可以表述为：

```text
在 SDNet baseline 的基础上，引入时空物理一致性增强后，模型在武汉地铁密集测试场景上的 MAE、MSE/RMSE、WRAE 三个核心评价指标均有下降，说明改进方法能够缓解相邻帧流入/流出预测与人数变化不一致的问题，提高视频级行人流量计数的稳定性。
```

### 6. 可视化结果

第三步已生成指标对比图和测试场景人数对比图：

```text
D:\模式识别与机器学习课设\MovingDroneCrowd-main\test_results\WuhanMetroCrowd\SDNet_wuhan_sdnet_temporal_refine_e30_640_val2test_4_best_model\metric_comparison.png
D:\模式识别与机器学习课设\MovingDroneCrowd-main\test_results\WuhanMetroCrowd\SDNet_wuhan_sdnet_temporal_refine_e30_640_val2test_4_best_model\scene_count_comparison.png
```

其中：

```text
metric_comparison.png：展示 baseline 与 refined 在 MAE、MSE/RMSE、WRAE 等指标上的对比。
scene_count_comparison.png：展示测试场景中 GT、baseline 预测人数、refined 预测人数的对比。
```

Baseline 的测试密度图/流量图可视化位于：

```text
D:\模式识别与机器学习课设\MovingDroneCrowd-main\test_results\WuhanMetroCrowd\SDNet_wuhan_sdnet_counter_e30_640_visual_test_4_best_model\11\0_0_visual.png
D:\模式识别与机器学习课设\MovingDroneCrowd-main\test_results\WuhanMetroCrowd\SDNet_wuhan_sdnet_counter_e30_640_visual_test_4_best_model\12\0_0_visual.png
```

---

## 三、当前进度总结

当前第（2）步已经完成：已将 SDNet baseline 在 WuhanMetroCrowd 上训练 30 epoch，保存了验证 checkpoint、最优模型和测试集评价结果。

当前第（3）步也已经完成到可以写入报告的程度：已实现训练阶段物理约束和时空一致性损失，完成了 5 epoch 物理约束微调和 1 epoch 时序监督微调实验，并记录了对应 epoch 指标。由于训练型微调结果未超过 baseline，最终采用推理阶段时空物理一致性增强作为有效改进方案。该方案在测试集上使 MAE 从 28.15 降至 26.43，MSE/RMSE 从 28.17 降至 26.44，WRAE 从 171.34 降至 162.21，说明改进后模型在密集视频行人流量计数任务上具有更好的视频级计数稳定性。
