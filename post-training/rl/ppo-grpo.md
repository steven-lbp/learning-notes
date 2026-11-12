# 策略梯度、PPO 与 GRPO

**强化学习先评价模型的回答，再调整参数，让模型更倾向于生成表现好的回答。**

**RL evaluates model responses, then adjusts parameters to favor better-performing responses.**

## 1. 策略概率是什么意思？

$\pi_\theta$ 表示模型的策略；$\pi_\theta(o\mid q)$ 表示这个模型面对问题 $q$ 时，生成回答 $o$ 的概率。

| 符号 | 含义 |
| --- | --- |
| $\pi$ | Policy，给定上下文时的输出概率分布 |
| $\theta$ | 模型参数 |
| $q$ | 输入问题或 prompt |
| $o$ | 模型生成的回答 |
| $\log\pi_\theta(o\mid q)$ | 生成这个回答的自然对数概率 |

例如生成某个回答的概率是 0.8，它的 log 概率约为 −0.223。**概率越大，对数概率也越大，越接近 0。**

如果 $o$ 是完整回答，整段回答的 log 概率就是每个 Token 的 log 概率之和：

$$
\log\pi_\theta(o\mid q)
=\sum_t\log\pi_\theta(o_t\mid q,o_{<t})
$$

每个 Token 的上下文包含问题和此前已经生成的内容。

$\pi_\theta$ is the model's policy, and $\pi_\theta(o\mid q)$ is the probability of generating response $o$ for prompt $q$.

| Symbol | Meaning |
| --- | --- |
| $\pi$ | Output probability distribution given the context |
| $\theta$ | Model parameters |
| $q$ | Input prompt |
| $o$ | Generated response |
| $\log\pi_\theta(o\mid q)$ | Natural logarithm of the response probability |

A probability of 0.8 has a log probability of about −0.223. **Higher probabilities have higher log probabilities, closer to 0.**

For a complete response, the log probability is the sum of token log probabilities, as shown above. Each token is conditioned on the prompt and all preceding tokens.

## 2. Advantage 如何影响概率？

**Advantage（$A$）表示这次回答相对 baseline 好多少。** 用单个样本的策略梯度损失理解更新方向：

$$
L=-A\log\pi_\theta(o\mid q)
$$

把 $A$ 看作本次更新的固定分数：

| Advantage | 含义 | 最小化 Loss 的方向 |
| --- | --- | --- |
| $A>0$ | 比基准好 | 提高这个回答的生成概率 |
| $A<0$ | 比基准差 | 降低这个回答的生成概率 |
| $A=0$ | 与基准相当 | 该项不提供梯度信号 |

当 $A>0$，概率越大 → log 概率越大 → 乘上负的 $A$ 后 Loss 越小。因此，训练会鼓励提高这个回答的概率。

**Advantage ($A$) measures how much better a response is than its baseline.** The single-sample policy-gradient Loss above shows the update direction when $A$ is treated as fixed for the update.

| Advantage | Meaning | Direction favored by minimizing Loss |
| --- | --- | --- |
| $A>0$ | Better than baseline | Increase the response probability |
| $A<0$ | Worse than baseline | Decrease the response probability |
| $A=0$ | Equal to baseline | No gradient signal from this term |

For positive $A$, increasing the probability increases its log probability. Multiplication by negative $A$ then lowers Loss, encouraging the model to increase that response's probability.

## 3. Loss 怎么变成参数更新？

先对 Loss 求梯度，再由优化器更新参数。以普通 SGD 为例：

$$
\begin{aligned}
\nabla_\theta L
&=-A\nabla_\theta\log\pi_\theta(o\mid q)\\
\theta_{\mathrm{new}}
&=\theta_{\mathrm{old}}-\eta\nabla_\theta L\\
\Delta\theta
&=\eta A\nabla_\theta\log\pi_\theta(o\mid q)
\end{aligned}
$$

- **$\nabla_\theta$：** 对模型参数求梯度。$\nabla_\theta\log\pi_\theta$ 给出提高该回答 log 概率的局部方向。
- **$\eta$：** 学习率，决定沿这个方向走多远。
- **$\Delta\theta$：** 新参数减旧参数，即 $\theta_{\mathrm{new}}-\theta_{\mathrm{old}}$。
- **$\propto$：** 成正比。写 $\Delta\theta\propto A\nabla_\theta\log\pi_\theta$，是在省略正的学习率，突出方向关系。

**A 决定顺着还是逆着这个方向更新，梯度决定参数怎样变化。** Adam 等优化器还会结合梯度历史调整实际更新。

Differentiate Loss with respect to the parameters, then let the optimizer apply the update. The equations above use ordinary SGD.

- **$\nabla_\theta$** takes the gradient with respect to model parameters. The log-probability gradient points in the local direction that increases the response's log probability.
- **$\eta$** is the learning rate and controls step size.
- **$\Delta\theta$** is the new parameter vector minus the old one.
- **$\propto$** means proportional to; it omits the positive learning rate to emphasize the direction.

**Advantage determines whether to follow or oppose the log-probability gradient; the gradient determines how parameters change.** Optimizers such as Adam also use gradient history.

## 4. PPO 与 GRPO 的核心区别

两者都使用 Advantage 指导策略更新，关键区别是 **baseline 从哪里来**。

| 维度 | PPO | GRPO |
| --- | --- | --- |
| 基准来源 | 学习一个 Value 函数，估计当前状态的预期回报 | 同一问题下，一组回答的平均奖励 |
| 怎样理解 Advantage | 实际表现相对预期好多少 | 这个回答相对同组回答好多少 |
| 额外的 Value / Critic | 通常需要 | 不需要 |
| 学生考试的类比 | 和“预计能考的分数”比较 | 和“同组同学的平均分”比较 |

PPO 常结合 Value 函数和后续奖励来估计 Advantage；GRPO 则用组内相对得分，减少训练 Value 模型的显存与计算开销。代价是每道题需要采样一组回答进行比较。[PPO 论文](https://arxiv.org/abs/1707.06347)、[DeepSeekMath §4.1](https://arxiv.org/html/2402.03300v3#S4.SS1)

Both methods use Advantage to guide policy updates. The main distinction is **where the baseline comes from**.

| Aspect | PPO | GRPO |
| --- | --- | --- |
| Baseline | A learned Value function estimating expected return | Mean reward of responses to the same prompt |
| Advantage intuition | Performance relative to expectation | Performance relative to the group |
| Additional Value / Critic model | Usually required | Not required |
| Exam analogy | Compare with the expected score | Compare with classmates' mean score |

PPO often combines Value estimates with subsequent rewards. GRPO uses group-relative scores to save the memory and compute needed for a Value model, while requiring a group of sampled responses per prompt.

## 5. GRPO 怎样计算组内优势？

以最终结果奖励为例：

1. 对同一个问题采样一组回答。
2. 用规则、验证器或奖励模型打分。
3. 用组内均值作比较基准，并按标准差缩放。
4. 根据 Advantage 更新策略。

$$
A_i=\frac{R(o_i)-\mu_G}{\sigma_G+\epsilon}
$$

其中 $R(o_i)$ 是当前回答的奖励，$\mu_G$、$\sigma_G$ 是组内奖励的均值和标准差；$\epsilon$ 用于避免分母为零。

例如同一道题的回答分别得 1、0、1 分，两个 1 分回答的 Advantage 为正，0 分回答为负。

For outcome rewards:

1. Sample a group of responses to one prompt.
2. Score them with rules, verifiers, or a reward model.
3. Subtract the group mean and scale by the standard deviation.
4. Update the policy using the resulting Advantage.

In the formula above, $R(o_i)$ is the response reward; $\mu_G$ and $\sigma_G$ are the group mean and standard deviation. $\epsilon$ avoids division by zero.

For rewards 1, 0, and 1, the two successful responses have positive Advantage and the unsuccessful response has negative Advantage.

## 6. 更新时怎样控制变化？

| 机制 | 控制什么 |
| --- | --- |
| PPO / GRPO 的 clipping | 根据新旧策略的 Token 概率比，裁剪目标中继续推动某些变化的收益 |
| KL 惩罚 | 约束当前策略与参考策略的分布差异；参考模型常取训练初始的 SFT 模型 |
| 梯度裁剪 | 在优化器更新前限制梯度范数 |

**旧策略**是生成当前这批 rollout 的策略版本；**参考策略**是 KL 对齐的基准。两者在训练中承担不同角色。[DeepSeekMath §4.1.1](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS1)

| Mechanism | What it controls |
| --- | --- |
| PPO / GRPO clipping | Limits the objective's incentive for certain changes, using new-to-old token probability ratios |
| KL penalty | Constrains distribution differences from a reference policy, often the initial SFT model |
| Gradient clipping | Limits the gradient norm before the optimizer update |

The **old policy** generated the current rollout batch. The **reference policy** provides the KL baseline. They have different roles during training.
