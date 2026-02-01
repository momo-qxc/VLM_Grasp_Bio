# 架子抓取后乱旋转问题记录

## 现象
- 在架子抓取场景下，抓住物体后末端会突然自旋/抖一下，退出货架时动作不稳定。

## 根因
- 闭合/松开夹爪的循环里，`env.step(action)` 时 **`action[:6]`（关节指令）未赋值，保持默认全 0**。
- 这等价于在夹爪动作期间同时给了“手臂回零位”的指令，导致末端突然被拉回零位方向，出现自旋/抖动。

## 修复思路
- 在夹爪动作的每一步都“锁住当前关节”，只修改夹爪通道 `action[-1]`。
- 架子抓取流程恢复为：退出货架 → 平移到放置点上方 → 下降放置，全程保持姿态不变。

## 关键代码对比

### 闭合夹爪前（旧）
```79:86:grasp_process_optimized.py
    # 闭合夹爪抓取
    for i in range(1000):
        action[-1] += 0.2
        action[-1] = np.min([action[-1], 255])
        env.step(action)
```

### 闭合夹爪后（新）
```79:87:grasp_process_optimized.py
    # 闭合夹爪抓取
    # 保持手臂关节不动，只动夹爪，避免被 action[:6]=0 拉回零位
    for i in range(1000):
        action[:6] = robot.get_joint()
        action[-1] += 0.2
        action[-1] = np.min([action[-1], 255])
        env.step(action)
```

### 松开夹爪前（旧）
```961:965:grasp_process_optimized.py
    for i in range(1000):
        action[-1] -= 0.2
        action[-1] = np.max([action[-1], 0])
        env.step(action)
```

### 松开夹爪后（新）
```961:967:grasp_process_optimized.py
    for i in range(1000):
        action[:6] = robot.get_joint()
        action[-1] -= 0.2
        action[-1] = np.max([action[-1], 0])
        env.step(action)
```

### 架子抓取退出/放置（新要点）
- 退出货架后继续抬高一点（`planner4` + `planner_lift`）。
- 以当前真实位姿为起点，平移到 `target_pos` 上方（保持 `grasp_rotation` 姿态）。
- 下降到放置高度，再松开夹爪，同样锁住关节指令。

## 总结
- 问题核心是：夹爪动作时必须显式写入 `action[:6] = robot.get_joint()`，否则默认为 0 会把手臂拉回零位，引发乱旋转。
- 修复后，架子抓取能够平稳退出并放置到目标位置。若需调整放置高度/位置，修改 `target_pos` 和下降高度即可。 

