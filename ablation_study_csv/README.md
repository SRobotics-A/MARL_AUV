# 消融实验结果数据

## action_ablation_csv
动作空间/动作表示的消融数据。典型例子：

- 直接学力/力矩 vs 学期望加速度/角速度（再接控制器）
- 是否加入动作平滑/动作约束
   输出一般会含 reward、成功率、姿态误差、能耗等随时间/训练步数变化的曲线数据。

## centralized_critic_csv

“集中式 critic/集中式训练（CTDE）”相关的消融。典型例子：

- centralized critic vs decentralized critic
- critic 输入是否包含全局信息（其他无人机状态/载荷全局状态）

## history_ablation_csv
“是否使用历史信息/时间堆叠（frame stacking）/RNN”等消融。典型例子：

- 只用当前观测 vs 加 k 步历史
- MLP vs GRU/LSTM
   用来验证部分可观测（POMDP）下历史信息的收益。

## obs_ablation_csv
观测项的消融。典型例子：

- full obs vs partial obs（你之前 README 里也提到 partial_obs）
- 去掉某些观测（如载荷角速度、绳索角度、邻机相对位姿等）
