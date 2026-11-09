# H2R Phase 1 imitator 适配说明

本文仅比较通用单手 imitator 预训练。Upstream 基准为 `ManipTrans/ManipTrans@a3d08cf`；H2R 为当前 `virtual-wrist-position-v3 / training_revision=7`。

Upstream 参数取 `main/cfg/config.yaml`、`DexHandImitator.yaml` 和 `DexHandImitatorPPO.yaml` 合并后的有效默认值，不采用 README 中其他任务的命令行覆盖。两列使用相同计数口径；实际运行覆盖值以产物 recipe 为准。

## 1. 方法主体与网络

| 项目 | Upstream | H2R |
| --- | --- | --- |
| PPO 与策略 | `PPOAgent`、字典观测网络、连续高斯策略、输入／价值 normalizer | 复用原生实现 |
| 特征编码器 | `SimpleFeatureFusion`：3 层、宽度 512、输出 256、Swish | 相同 |
| Actor/critic | 共享网络；MLP `[256, 512, 128, 64]`，ELU | 相同 |
| 策略标准差 | 初始 logstd=0；`fixed_sigma=True` 表示与状态无关的可学习标准差 | 相同 |
| 跟踪目标 | 下一帧人体手部关键点与腕状态 | 保留人体目标，Phase 1 不做手指 IK，不以机器人 FK 替代人体目标 |

## 2. 机器人、物理、动作与观测

表中 `a` 为原生 soft clamp 后的动作，`q_actual(t)` 为控制步开始时的实测腕关节位置。

| 项目 | Upstream | H2R |
| --- | --- | --- |
| 机器人资产 | 原生 URDF 手，默认 `inspire` | 从共同 task 的 MJCF 抽取 Sharpa 单手，继承实际增益、惯量、碰撞及限位 |
| 物理后端 | IsaacGym / PhysX | mjlab 1.6 `Simulation` / MJWarp |
| 时间步 | `dt=0.0166667`，2 substeps | 控制 60 Hz、物理 120 Hz、decimation=2 |
| 手腕表示 | 自由浮动根 | 6 个有限位虚拟关节；旋转使用连续 Euler 分支 |
| 腕执行器 | 默认直接施加外力／力矩，另有可选 PID；线性／角阻尼均为 20 | 使用共同位置执行器及其增益、阻尼；不新增外部 wrench 或速度补偿 |
| 腕动作解码 | 力为 `a_xyz × dt × 1.0 × 500`，力矩为 `a_rot × dt × 0.1 × 200` | `q_target = q_actual(t) + scale × a`；平移 scale=0.12 m，旋转 scale=1.0 rad |
| 动作更新与保持 | 每控制步更新 wrench，由 PhysX 执行子步 | 每控制步读取实际 q 并计算目标，在两个物理子步内保持；不累加上一目标、不另乘 dt、不回绕腕角 |
| 零腕动作 | 不施加策略腕力／力矩，物理阻尼仍起作用 | 以本步实测腕位置为目标，由位置执行器产生保持／阻尼作用 |
| 参考轨迹进入控制的位置 | 参考用于观测、奖励及 RSI；默认腕 wrench 不直接叠加参考 | 参考仍用于观测、奖励及 RSI；v3 腕目标不叠加参考轨迹 |
| 动作尺度的物理含义 | 60 Hz 下默认最大分量约为 8.33 N、0.333 N·m | 0.12 m／1.0 rad 是目标偏置，不是每步实际位移，也不等价于上游力幅值；控制步开始时近似有 `τ = Kp × scale × a − Kd × dq` |
| 执行器参数来源 | 原生手模型及环境配置 | 封存 task；例如平移／旋转 `Kp=300/100`、`Kd≈51.92/20`，方法不修改这些参数。该例按上游力矩直接换算仅得约 0.00333 rad，不能视作等效运动能力 |
| 手指动作 | soft clamp 后映射到手指关节范围 | 保留映射，另遵守启用的 actuator ctrl range |
| 重力与场景 | 原生 imitator 场景，手部 `disable_gravity=True` | 沿用共同 task 的重力和手部 gravcomp；抽取单手时移除物体、桌面和另一只手 |
| 域随机化 | imitator 默认 `randomize=False` | Phase 1 不启用域随机化 |
| 接触容量 | PhysX `max_gpu_contact_pairs=1048576`，为全局配置 | 单手每环境 contact / constraint capacity=128 / 1024；溢出报错，两者计数口径不同 |
| 基础观测 | 手指 q、cos(q)、sin(q)、dq、腕状态及人体目标位置／速度差 | 保留这些特征；privileged 追加腕 dq，proprioception 追加腕 q，target 追加下一帧参考腕 q |
| 动作／观测维度 | 随原生机器人变化；默认腕动作 6 维，加手指 DOF | Sharpa 动作 28 维（22 手指 + 6 腕）；privileged / proprioception / target 为 28 / 85 / 209 维 |
| 观测裁剪 | ±5 | ±5 |

## 3. 数据与参考构造

| 项目 | Upstream | H2R |
| --- | --- | --- |
| 数据入口 | `dataIndices` 经 `ManipDataFactory` 选择原生 loader | 独立 hand-motion corpus，来源为 ARCTIC、OakInk2、HOT3D、TACO 的可用录制；不受 manipulation 任务清单限制，不表示覆盖官方全集 |
| 最短片段筛选 | 原生加载路径没有此统一筛选步骤 | 方法侧排除少于 4 源帧或不足 0.5 s 的片段，时长容差 `1e-10 s`；记录 `clip_selection.json` |
| 序列分配 | 建环境时按 `i % len(dataIndices)` 分配示范 | 每次 reset 重新采样窗口 |
| 数据集／左右来源权重 | 由 `dataIndices` 的内容及分配次数决定 | 四数据集等概率、源左右侧等概率；池内按有效转移时长加权 |
| 真实／合成比例 | 所选原生加载路径没有此固定混合配方 | 90% 真实、10% MANO 端点旋转 SLERP 合成；合成段在源片段允许范围内取 0.5–5 s |
| 镜像选择 | 数据索引的 `M` 后缀区分镜像数据类型 | 源侧与目标手不同时自动镜像 |
| 时间与姿态重建 | 各 loader 构造原生参考；基础速度计算使用固定采样间隔，默认 `skip=2` 对应 60 Hz | MANO 在 60 Hz 解码／插值并恢复真实腕轨迹；核验 MANO 资产与源关键点重建 |
| 速度 | 固定时间间隔差分，Gaussian 平滑 sigma=2 | 按实际采样时间差分，保留 sigma=2；虚拟腕关节速度另经 Jacobian 换算 |
| 长序列 | 超过 2000 帧时截断 | 先计算完整参考，再切成至多 2000 样本的腕限位可行窗口；相邻窗口共享端点，全部源转移恰好覆盖一次 |
| 腕角连续性与非法参考 | 自由根不需要满足虚拟腕关节的 Euler 分支限位 | 每窗固定等价分支和 `2π` 偏移，episode 内不换圈；不能表示的参考报错，不扩大共同限位 |
| 插值端点误差 | 不使用这套 H2R MANO 插值器 | 仅将不超过 1 ns 的舍入越界归位，真正外推仍报错 |
| 参考内存布局 | 各环境选中的序列打包后送入训练 device | CPU mmap，加每环境 128+2 样本的 GPU 缓存，按需取数 |
| 准备并行 | 环境构建时按原生 loader 加载、打包示范 | 默认 8 个 CPU worker，每 worker 1 个计算线程；左右手共享 MANO 解码，按源顺序合并，不改变采样配方 |

## 4. 初始化、奖励与终止

| 项目 | Upstream | H2R |
| --- | --- | --- |
| RSI 起始帧 | `floor(0.99 × seq_len × U[0,1))` | 保留公式，再限制到 `length-2`，保证至少一次有效转移 |
| 手指初始位置 | 初始角 π/36；位置噪声标准差为关节范围的 1/8，按关节限位裁剪 | 相同 |
| 手指初始速度 | 高斯噪声标准差 0.1，再按原生 speed limit 裁剪 | 高斯噪声标准差 0.1，不额外执行原生 speed limit 裁剪 |
| 腕初始化噪声 | 平移标准差 0.01 m、旋转角标准差 π/18、线／角速度标准差 0.01 | 保留噪声参数；速度转换为虚拟腕关节速度 |
| 腕姿态噪声越界 | 自由根没有相同的虚拟腕限位重采问题 | 先核验参考及固定分支；最多重采 64 次，耗尽后回退到同一帧未扰动参考腕姿态，保留手指和速度噪声 |
| RSI 回退分布与记录 | 使用原生自由腕扰动分布 | 回退时取消腕姿态噪声，分布与上游不同；reset、拒绝和回退计数写入 `sampling.json` 并随 resume 保存；非法参考、非有限值及 FK 错误直接失败 |
| 奖励与阈值 | 原生 imitator 奖励权重及失败阈值 | 复用原生函数及参数 |
| 阈值课程 | `0.7 + 0.3 × (2e)^(-steps/128000)`，取 IsaacGym frame count | 保留公式，按控制调用计数，避免将两个物理子步计作两次课程推进 |
| 失败宽限 | 关键点误差保留 20 步宽限，速度异常不受宽限 | 保留规则，以本次动作之前已完成的控制步数判断 |
| 功率 | 手指关节力 × 速度；根部外力／力矩 × 线／角速度 | 手指规则保留；腕改为虚拟关节广义执行器力 × 关节速度，取最后一次物理解算的力与积分后速度 |
| 成功终止 | `progress_buf + 1 + 3 >= max_length` | 跟踪至窗口真实最后一帧；有限序列终止设 `time_outs=false`，单纯 PPO horizon 截断仍可 bootstrap |
| Episode 步数记录 | `info.total_steps = progress_buf`，包含 RSI 起始帧索引 | 分开记录源帧位置与实际 episode 控制步数，不将 RSI 起始帧计作已执行步数 |

## 5. PPO 参数

环境数均按单次训练统计；H2R 左右手分别启动训练。

| 参数 | Upstream | H2R |
| --- | --- | --- |
| 并行环境 | 8192 | 8192 |
| 最大 epochs | 9999999999999 | 9999999999999 |
| horizon / minibatch / mini epochs | 32 / 1024 / 5 | 32 / 1024 / 5 |
| learning rate / schedule / KL | 5e-4 / adaptive / 0.008 | 5e-4 / adaptive / 0.008 |
| gamma / GAE tau | 0.99 / 0.95 | 0.99 / 0.95 |
| PPO clip / grad norm | 0.2 / 1.0 | 0.2 / 1.0 |
| critic / entropy / bounds 系数 | 4 / 0 / 1e-4 | 4 / 0 / 1e-4 |
| 输入／价值／advantage 归一化 | 开启 | 开启 |
| mixed precision | 关闭 | 关闭 |
| seed / action moving average | 42 / 1.0 | 42 / 1.0 |
| score_to_win / early_stop_epochs | 10000 / 9999999999999 | 10000 / 9999999999999 |
| save_best_after / 保存周期 | 10 / 1000 epochs | 10 / 1000 epochs |
| 每 epoch 转移数／minibatch 更新数 | 262144 / 1280 | 262144 / 1280 |
| 有限 epoch 预算覆盖入口 | `max_iterations=N` | `--epochs N` |

Upstream PPO YAML 中 `max_epochs=10000`、`early_stop_epochs=500` 是 fallback，实际被全局配置覆盖。两侧默认均近乎不设 epoch 上限，仍保留原生停止条件；该上限本身不表示收敛。

## 6. 启动、产物、诊断与兼容

| 项目 | Upstream | H2R |
| --- | --- | --- |
| 运行入口与环境 | 原生 IsaacGym 环境，`main/rl/train.py` 配合 Hydra task 配置 | 方法仓库内的独立环境，`python -m maniptrans_h2r.pretrain` 提供 prepare、train、check 子命令；不导入 bench |
| 训练输入／产物 | 原生 dataset 与 checkpoint | 版本化 prepared 与左右手独立 imitator 产物，封存参考、物理、控制和训练配方身份 |
| 数值检查 | 原生训练与 checkpoint 保存逻辑 | 增加原始模型输出、梯度、更新后参数、normalizer 和 checkpoint 的非有限值检查 |
| Resume 内容 | 保存模型、归一化、优化器、epoch/frame、最佳奖励等状态 | 保留这些内容，额外保存学习率调度、各 RNG、采样／RSI 计数、课程控制步数及配方绑定；恢复时重置物理世界，不保证逐位连续 |
| 参考与控制身份 | 原生机器人及自由腕 wrench 契约 | `virtual-wrist-position-v3`、`hand-reference-v3`、`preparation_revision=2`（`wrist-feasible-v1`）、`training_revision=7` |
| 旧 checkpoint／resume | 按原生观测、动作和恢复逻辑使用 | 原生 wrench checkpoint 不能直接使用；旧 H2R base 必须重训，旧 revision 或不同 worlds 的 resume 拒绝续接；满足当前参考签名的 prepared 可复用 |
| 旧 H2R 部署 | 不适用 | v1/v2 部署仍按其 `q_ref(t_next) + delta` 契约解码；当前 v3 使用实测 q 加增量，避免静默改变旧模型含义 |
| 固定跟踪诊断 | 没有这套 H2R corpus 回调；原生测试模式的阈值 scale=1 | epoch 0、每 250 epochs、实际训练结束时诊断；seed=43、严格 scale=0.7；最终执行均值和随机策略诊断，覆盖提前停止，隔离训练 RNG |
| 诊断指标 | 原生奖励／训练日志 | 记录腕与关键点误差、功率、有效前缀和时域覆盖；`wrist_action_saturation` 为 soft clamp 后六维腕动作绝对值至少 .95 的比例 |
| 诊断结果用途 | 按原生训练／测试入口解释 | tracking diagnostic 与训练 corpus 重叠，仅衡量手部跟踪，不代表未见数据泛化或物体操作成功率 |
| 诊断兼容与生命周期 | 没有 mjlab 诊断环境的 stream 生命周期 | 独立 check 拒绝不兼容的控制配方；诊断环境复用同一 CUDA stream 包装，避免回收时注销仍在使用的句柄 |

完整启动命令见 [H2R 使用说明](h2r.md)。上述适配不保证跨后端动力学或同 seed 样本逐一相同，也不构成论文结果的精确复现或长期 PPO 收敛声明。

## 源码依据

| 内容 | Upstream | H2R |
| --- | --- | --- |
| 版本／目录 | [固定版本源码](https://github.com/ManipTrans/ManipTrans/tree/a3d08cfe3c3a5868a7f057533bcaf759c5af4705) | [适配实现目录](../maniptrans_h2r) |
| 配置与训练 | `main/cfg/config.yaml`、`main/cfg/task/DexHandImitator.yaml`、`main/cfg/rl_train/DexHandImitatorPPO.yaml`、`lib/rl/base.py` | `training.py`、`config/imitator.yaml` |
| 环境与控制 | `maniptrans_envs/lib/envs/tasks/dexhandimitator.py` | `imitator_env.py`、`mjlab_env.py`、`control.py`、`observations.py` |
| 数据与参考 | `main/dataset/factory.py`、`main/dataset/base.py` 及各 loader | `pretrain.py`、`pretrain_worker.py`、`hand_reference.py`、`motion.py` |
| 固定诊断 | 无对应 H2R 模块 | `diagnostic.py` |
