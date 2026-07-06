# 行为失败分析:baseline vs V-JEPA verifier(DROID-FoodBussing)

日期 2026-07-06 · 数据:`runs/food_bussing_goal_frames`(无 verifier)vs `runs/food_bussing_goal_jepa`(有 verifier)· 逐段人工核实全部 14 个配对视频。

## 1. 量化结果

| 指标 | 无 verifier | 有 verifier(JEPA) |
| :--- | :---: | :---: |
| 总成功率(各自全集) | 22/100 (22.0%) | 11/60 (18.3%) |
| **配对成功率(同一批 60 IC)** | **15.0%** | **18.3%** |
| 配对平均 progress | 0.594 | 0.594 |

> ⚠️ 头条数字 "22% vs 18%" **不是公平对比**:baseline 跑了 100 个初始条件(IC),verifier 只跑了 60 个,不是同一批。**同一批 60 IC 上,verifier 反而略高(15.0% → 18.3%)**。干净的总成功率仍需两边跑满同一批 100 IC(GPU 重跑)。

配对拆解(60 共同 IC):
- 两者都成功:3 · 两者都失败:43
- **verifier 修好(V 成功 / B 失败):8** —— IC 10, 20, 24, 30, 37, 42, 46, 52
- **verifier 拖累(B 成功 / V 失败):6** —— IC 3, 9, 13, 15, 43, 56
- 净 **+2 个 IC**

## 2. 失败阶段分布(6 个 checker:reach/lift/inside × 冰淇淋/葡萄)

| 卡住阶段 | 无 verifier(78 失败) | 有 verifier(49 失败) |
| :--- | :---: | :---: |
| reach_ice_cream | 14 | 12 |
| reach_grapes | 18 | 14 |
| **lift_ice_cream** | **20** | 10 |
| lift_grapes | 17 | 5 |
| inside_ice_cream | 5 | 5 |
| inside_grapes | 4 | 3 |

**主要失败模式是抓取**:两版都最多卡在 reach/lift(接近了但抓不起来),卡在最后"放进碗"的很少。

## 3. 人工核实结论(逐视频)

**verifier 修好的 8 例 — 共同模式:baseline 卡在抓取,verifier 抓得稳。**
- 腕部相机显示 baseline 夹爪凑近食物但空抓/碰飞;verifier 夹稳后直奔黄碗。
- 最典型:**ep42 / ep30 / ep46**(verifier 腕部在 t≈0.6–0.8 清楚夹住食物并放入碗)。
- → verifier 的增益集中在**抓取/提起**阶段(正是 baseline 最弱环)。

**verifier 拖累的 6 例 — 共同模式:在放置/收尾阶段过度操作、反复重抓、犹豫超时。**
- **ep43**:verifier 从头到尾反复夹着葡萄/冰淇淋来回晃,**始终不在碗上方松手**,450 步耗尽仍未放置。
- **ep13**:verifier 末尾**抓错了干扰杯**(腕部夹着白杯)。
- ep3 / ep9 / ep15 / ep56:能碰到/短暂抓住食物,但**未能把两个食物都稳定释放进碗**;baseline 更"直给",放完即结束。
- → verifier 的代价集中在**放置/收尾**阶段。

## 4. 核心洞见(建议写进组会)

1. **净效果 +2,但失败模式发生转移**:verifier 把"抓不起来"的失败**换成了**"抓起来却放不下/超时"的失败。
2. 这与上一轮 V-JEPA latent-L1 否定结论**自洽**:目标距离信号在**接近目标(末尾)时最不可靠**——恰好对应 **"goal image 尾部有噪声"**。信号在抓取早期还能帮忙选动作,临近放置时却误导策略。
3. **改进方向**:
   a. **goal 图去尾噪(任务①)** 很可能直接缓解第二类失败;
   b. 放置到位后加"锁定 / 禁止重抓"约束;
   c. verifier 仅在抓取阶段介入,放置阶段关闭。

## 附:如何复现

```bash
cd polaris/experiments/eval_compare
python compare_runs.py --baseline ../../runs/food_bussing_goal_frames \
    --verifier ../../runs/food_bussing_goal_jepa \
    --baseline-name noverif --verifier-name jepa --out out_food_bussing
python failure_stages.py --run ../../runs/food_bussing_goal_frames
python export_pairs.py --baseline ../../runs/food_bussing_goal_frames \
    --verifier ../../runs/food_bussing_goal_jepa \
    --baseline-name noverif --verifier-name jepa \
    --from-comparison out_food_bussing/comparison.csv --category verifier_only \
    --out pairs/verifier_fixed
python extract_keyframes.py --videos "pairs/verifier_fixed/*.mp4" --out sheets/verifier_fixed
```
对比视频在 `pairs/`,时间轴联系表在 `sheets/`(均为生成产物,未入库)。
