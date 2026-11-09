# H2R Phase 1 imitator 适配说明

本文仅记录通用单手 imitator 预训练。比较基准：上游 `ManipTrans/ManipTrans@a3d08cf`；H2R 当前实现为 `virtual-wrist-position-v3 / training_revision=7`。H2R 分别训练左右手，适配对象为 Sharpa Wave。

上游参数指 `main/cfg/config.yaml`、`DexHandImitator.yaml` 和 `DexHandImitatorPPO.yaml` 合并后的有效默认值，不采用 README 中其他任务的命令行覆盖。实际运行的覆盖参数以产物中的 recipe 为准。

## 1. 保留的方法主体

- 复用原生 `PPOAgent`、字典观测网络、连续高斯策略、输入／价值 normalizer，以及 imitator 奖励和阈值课程。
- 特征编码器仍为 `SimpleFeatureFusion`：3 层、宽度 512、输出 256、Swish；后接 `[256, 512, 128, 64]`、ELU，actor/critic 共享网络。
- 初始 logstd 为 0；`fixed_sigma=True` 表示与状态无关的可学习标准差，并非冻结标准差。
- 保留 sigmoid soft clamp、手指位置目标控制及下一帧人体手部跟踪目标。Phase 1 不做手指 IK，不把机器人 FK 当作人体目标。

## 2. 机器人、物理与控制

| 项目 | 上游 | H2R 适配 |
| --- | --- | --- |
| 物理后端 | IsaacGym / PhysX | mjlab 1.6 `Simulation` / MJWarp |
| 时间步 | `dt=0.0166667`、2 substeps | 控制 60 Hz、物理 120 Hz、decimation=2 |
| 机器人 | 原生 URDF 手，默认 `inspire` | 从共同 task 的 MJCF 抽取 Sharpa 单手；继承实际增益、惯量、碰撞、限位及物理配置 |
| 手腕 | 自由浮动根；默认外力／力矩，另有可选 PID；线性／角阻尼均为 20 | 6 个有限位虚拟关节，使用共同位置执行器及其阻尼 |
| 腕动作 | 默认力为 `a_xyz × dt × 1.0 × 500`，力矩为 `a_rot × dt × 0.1 × 200` | `q_actual(t) + scale × a`；平移 scale 为 0.12 m，旋转为 1.0 rad |
| 手指动作 | soft clamp 后映射关节范围 | 保留映射，另遵守启用的 actuator ctrl range |
| 重力／场景 | 手部 `disable_gravity=True` | 沿用共同 task 的重力及手部 gravcomp 设置；移除物体、桌面和另一只手 |
| 接触容量 | 原生 PhysX 配置 | 单手 contact / constraint capacity 为 128 / 1024，溢出报错 |

H2R Phase 1 不启用域随机化。上述变化不保证 PhysX 与 MJWarp 动力学等价；原生自由腕 checkpoint 不能直接用于虚拟腕版本。

腕目标以每个控制步开始时的实测关节位置为基准，在两个物理子步内保持；不累加上一目标，不乘 dt，不叠加参考轨迹，不对腕角回绕。零动作以当前腕位置为目标，参考仍用于观测、奖励和 RSI。v1/v2 的 `q_ref(t_next) + delta` 会在零动作时主动跟踪参考；v3 去掉了这一前馈，让策略产生腕运动。

增量尺度针对共同执行器选定，不是上游力幅值的换算，也不表示每步实际移动 0.12 m 或旋转 1 rad。以 `Kp=300/100`、`Kd≈51.92/20` 的平移／旋转配置为例，当前位置加增量的初始作用力近似为 `Kp × scale × a − Kd × dq`；照搬上游力幅值会使旋转增量仅约 0.00333 rad，无法沿用原先参考前馈提供的运动能力。具体增益仍取封存 task，不由方法修改。

观测保留手指 `q/cos(q)/sin(q)`、腕状态、人体关键点及速度差，追加虚拟腕 `dq` 到 privileged、`q` 到 proprioception、下一帧参考腕 `q` 到 target。Sharpa 的动作维度为 28（22 手指 + 6 腕），三组观测维度为 28 / 85 / 209；观测仍裁剪到 ±5。

## 3. 数据与参考构造

上游按 `dataIndices` 加载原生 dataset，并以 `i % len(dataIndices)` 将示范分配给环境；超过 2000 帧的序列截断。H2R 改为独立的通用手部 corpus 和 reset 时采样：

- 来源为 ARCTIC、OakInk2、HOT3D、TACO 的可用录制，不受 manipulation 任务清单限制；不表示覆盖各数据集官方全集。
- 方法侧排除少于 4 源帧或不足 0.5 s 的片段（时长容差 `1e-10 s`），将选择结果写入 `clip_selection.json`。
- 四数据集等概率、源左右侧等概率；90% 真实片段、10% MANO 端点旋转 SLERP 合成片段。异侧来源镜像到目标手；池内按有效转移时长加权。
- MANO 在 60 Hz 解码／插值，并恢复真实腕轨迹；合成段时长在源片段允许范围内取 0.5–5 s。速度按真实采样时间计算并进行 Gaussian 平滑（sigma=2），虚拟腕速度经 Jacobian 换算。
- 完整片段先计算连续参考，再切成至多 2000 样本的腕限位可行窗口。相邻窗口共享端点，全部源转移恰好覆盖一次；每窗固定 Euler 分支及 `2π` 偏移，episode 内不换圈。无法表示的参考报错，不扩大共同限位。
- 插值端点仅对不超过 1 ns 的浮点舍入越界归位，真正的外推仍报错。

这些是数据分布和参考表示的适配，不应视为与上游训练集完全一致。CPU mmap 与每环境 GPU 小缓存替代全集驻留显存；准备默认 8 个 CPU worker、每 worker 1 个计算线程，左右手共享 MANO 解码。这些优化不改变采样配方。

## 4. 初始化、奖励与终止

RSI 保留 `floor(0.99 × seq_len × U[0,1))`，H2R 再限制到 `length-2`，保证至少一次有效转移。沿用手指初始角 π/36、位置噪声标准差为关节范围的 1/8、速度噪声标准差 0.1；上游另对手指速度执行原生 speed limit 裁剪。腕平移噪声标准差为 0.01 m，旋转噪声角标准差为 π/18，线／角速度噪声标准差为 0.01。

受限虚拟腕新增 RSI 规则：先核验未扰动参考及固定分支；姿态噪声越界最多重采 64 次，耗尽后回退到同一帧未扰动参考腕姿态，保留手指和速度噪声。非法参考、非有限值或 FK 错误直接失败。回退会取消腕姿态噪声，因而与上游自由腕分布不同；reset、拒绝和回退计数写入 `sampling.json` 并随 resume 保存。

奖励权重与阈值沿用上游，阈值课程为 `0.7 + 0.3 × (2e)^(-steps/128000)`，`steps` 按控制调用累计。关键点失败保留 20 步宽限，速度异常不受此宽限；奖励使用积分后的状态。功率仍按力与速度计算，但原生根部 wrench 功率改为虚拟腕广义执行器力与关节速度的功率。

上游在 `progress_buf + 1 + 3 >= max_length` 时成功结束；H2R 跟踪至窗口真实最后一帧，将其视为有限序列终止（`time_outs=false`）。单纯 PPO horizon 截断仍可 bootstrap。H2R 分开记录源帧位置与实际 episode 控制步数，不将 RSI 起始帧计作已执行步数。

## 5. PPO 参数对照

| 参数 | 上游有效默认值 | H2R 默认值 |
| --- | --- | --- |
| 并行环境 | 8192 | 每手 4096 |
| 最大 epochs | 9999999999999（全局覆盖） | 10000 |
| horizon / minibatch / mini epochs | 32 / 1024 / 5 | 相同 |
| learning rate / schedule / KL | 5e-4 / adaptive / 0.008 | 相同 |
| gamma / GAE tau | 0.99 / 0.95 | 相同 |
| PPO clip / grad norm | 0.2 / 1.0 | 相同 |
| critic / entropy / bounds 系数 | 4 / 0 / 1e-4 | 相同 |
| 输入／价值／advantage 归一化 | 开启 | 相同 |
| mixed precision | 关闭 | 相同 |
| seed / action moving average | 42 / 1.0 | 相同 |
| score_to_win / early_stop_epochs | 10000 / 9999999999999 | 相同 |
| save_best_after / 保存周期 | 10 / 1000 epochs | 相同 |

H2R 每手名义预算为 `4096 × 32 × 10000 = 1,310,720,000` 次转移，仍可能触发原生提前停止。上游 PPO YAML 中 `max_epochs=10000`、`early_stop_epochs=500` 是 fallback，不能忽略全局覆盖后直接用于比较。相同 seed 不保证跨后端／随机数实现的逐样本一致。

## 6. 启动、产物与诊断

H2R 新增独立环境及 `python -m maniptrans_h2r.pretrain prepare|train|check` 入口；训练实现留在方法仓库，不导入 bench。左右手分别输出模型。完整命令和环境安装见 [H2R 使用说明](h2r.md)。

- 新增模型／梯度／normalizer 非有限值检查，以及模型、优化器、学习率状态、随机数状态、采样计数和课程步数的 resume 保存。恢复时重置物理世界，不保证与不中断运行逐位一致。
- 新增 epoch 0、每 250 epochs 和实际训练结束时的 tracking diagnostic；seed=43、严格阈值 scale=0.7。最终执行均值和随机策略诊断，也覆盖提前停止；诊断隔离训练 RNG。新增 `wrist_action_saturation`，记录 soft clamp 后腕动作绝对值至少 .95 的比例。
- 诊断与训练 corpus 重叠，只衡量手部轨迹跟踪，不代表未见数据泛化或物体操作成功率。
- 当前身份为 `virtual-wrist-position-v3`、`hand-reference-v3`、`preparation_revision=2`（`wrist-feasible-v1`）、`training_revision=7`。已有 prepared 可复用；旧 base 必须重新训练，旧 resume 不跨版本续接。独立 check 拒绝控制配方不兼容的模型，旧 deployment 保留原解码方式。

受限腕噪声重采回退、提前停止后的最终诊断，以及诊断环境共享 CUDA stream 包装，均属于 H2R 迁移层逻辑；对应的重采耗尽和 stream 回收问题不是上游自由腕实现已有的问题。上述适配不构成论文结果的精确复现或长期 PPO 收敛声明。

## 源码依据

- 上游：[固定版本源码](https://github.com/ManipTrans/ManipTrans/tree/a3d08cfe3c3a5868a7f057533bcaf759c5af4705)；有效参数来自 `main/cfg/config.yaml`、`main/cfg/task/DexHandImitator.yaml` 和 `main/cfg/rl_train/DexHandImitatorPPO.yaml`，环境为 `maniptrans_envs/lib/envs/tasks/dexhandimitator.py`，数据加载为 `main/dataset/base.py`。
- H2R：[适配实现目录](../maniptrans_h2r)；主要入口为 `pretrain.py`，参考为 `hand_reference.py` / `motion.py`，环境为 `imitator_env.py` / `mjlab_env.py`，控制为 `control.py`，训练与参数为 `training.py` / `config/imitator.yaml`，诊断为 `diagnostic.py`。
