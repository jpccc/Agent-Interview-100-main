# 评估方法论：从 LLM 评估到 Agent 评估

> 难度：基础
> 分类：Evaluation

## 简短回答

**LLM 评估三大类：**
1. **自动指标评估** — 用算法计算的确定性指标（如 BLEU、ROUGE、精确匹配），速度快、成本低，但只能衡量表面特征
2. **人工评估** — 由人类标注者评判输出质量，是“金标准”但成本高、难以规模化
3. **LLM-as-Judge** — 用强 LLM 评估其他 LLM 的输出，顶级 Judge（GPT-4o / Claude Opus）与人工评估的 Cohen's Kappa ≈ 0.78-0.84，逼近人类-人类一致性（κ≈0.80），是近年主流趋势

> ⚠️ 注意 percent agreement 容易虚高（κ=0.62 也能 >80% 一致率），学术界（“Judging the Judges” arXiv:2406.12624）建议用 Cohen's Kappa 才是 Judge 可靠性的正确指标。

**生产推荐混合方案：** 自动指标做初筛，LLM-as-Judge 做质量评估，人工评估做最终校准。

**Agent 评估的根本性扩展：** LLM 评估关注单次输入输出的质量，而 Agent 评估关注 **多步决策轨迹的整体表现**——不仅看最终结果，还要评估推理过程、工具使用、规划质量和错误恢复能力。核心差异在于：Agent 涉及 LLM + 工具 + 环境的交互链，非确定性更强，评估维度也从文本质量扩展到任务完成能力、轨迹效率和行为安全。

## 详细解析

### 一、LLM 评估方法全景

| 维度 | 自动指标 | LLM-as-Judge | 人工评估 |
|------|---------|-------------|----------|
| **代表方法** | BLEU/ROUGE、精确匹配、F1 Score、Perplexity | GPT-4 打分、多维度评估、Pairwise 对比、Rubric 评分 | 专家评审、众包标注、用户反馈、A/B 测试 |
| **速度** | 最快 | 中等 | 最慢 |
| **成本** | 最低 | 中等 | 最高 |
| **质量** | 有限 | 较好 | 最好 |
| **规模** | 无限 | 大 | 小 |

Sebastian Raschka 将 LLM 评估总结为四种方法：多选基准、人类偏好、自动化 LLM 评估、和编程基准。

### 二、自动指标评估

自动指标的本质是用算法比较“模型的回答”与“标准答案”之间的**文字重叠程度**，速度快、成本低，但只看表面特征，不懂语义。

| 指标 | 核心问题 | 计算逻辑 | 适用场景 |
|------|---------|---------|----------|
| **精确匹配** (Exact Match) | 两个字符串一模一样吗？ | `prediction == reference` | 数学题答案、事实性问题、代码输出 |
| **BLEU** (precision-oriented) | 你说的话有多少是对的？ | prediction 中有多少 n-gram 出现在 reference | 机器翻译（怕你乱加东西） |
| **ROUGE** (recall-oriented) | 标准答案的信息你找回了多少？ | reference 中有多少 n-gram 出现在 prediction | 文本摘要（怕你漏掉关键信息） |
| **F1 Score** | 精确又完整吗？ | 2 × precision × recall / (precision + recall) | 信息提取（既要精确又要完整） |

**具体例子**：标准答案 `"北京是中国的首都"`，模型回答 `"北京是首都"`

- **精确匹配** → 0 分（不完全相同）
- **BLEU** → 较高（"北京""是""首都" 都在答案里，precision 高；但不管你有没有漏掉"中国的"）
- **ROUGE** → 约 50%（答案 6 个词，你只覆盖了 3 个，recall 低）
- **F1** → precision 和 recall 的调和平均，兼顾两侧

> **BLEU vs ROUGE 的本质区别：** 翻译任务容忍不了你“瞎编” → 用 precision-oriented 的 BLEU；摘要任务容忍不了你“漏掉重点” → 用 recall-oriented 的 ROUGE。两类任务对“漏掉信息 vs 多说信息”的容忍度不同，所以选用不同方向的指标。

**ROUGE 的变体**：
- **ROUGE-1**：unigram 召回率（逐词比较）
- **ROUGE-L**：最长公共子序列（衡量句子级流畅度，允许跳词但不允许乱序）

> ⚠️ **自动指标的共同局限：** `"北京是中国的首都"` vs `"中国的首都是北京"` — 语义完全相同，但 BLEU/ROUGE 可能不是满分（词序不同）。这些指标**无法评估回答的有用性、创造性、安全性**，LLM 时代参考价值有限。

<details>
<summary>📝 代码实现参考</summary>

```python
# 1. 精确匹配
def exact_match(prediction, reference):
    return prediction.strip() == reference.strip()

# 2. BLEU
from nltk.translate.bleu_score import sentence_bleu
score = sentence_bleu([reference.split()], prediction.split())

# 3. ROUGE
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
scores = scorer.score(reference, prediction)

# 4. F1 Score
def token_f1(prediction, reference):
    pred_tokens = set(prediction.split())
    ref_tokens = set(reference.split())
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(ref_tokens) if ref_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return f1
```
</details>

### 三、LLM-as-Judge

```python
async def llm_as_judge(question, answer, reference=None):
    """用 LLM 评估回答质量"""

    # 方式 1：直接评分（Pointwise）
    pointwise_prompt = f"""
    请评估以下回答的质量（1-5分）：

    问题：{question}
    回答：{answer}
    {"参考答案：" + reference if reference else ""}

    评分维度：
    - 准确性 (1-5)：事实是否正确？
    - 完整性 (1-5)：是否全面回答了问题？
    - 有用性 (1-5)：对提问者是否有帮助？
    - 清晰度 (1-5)：表达是否清楚？

    请给出每个维度的分数和简要理由，最后给出总分。
    """

    # 方式 2：对比评分（Pairwise）
    pairwise_prompt = f"""
    问题：{question}

    回答 A：{answer_a}
    回答 B：{answer_b}

    哪个回答更好？请从准确性、完整性和清晰度三个维度比较。
    输出：A 更好 / B 更好 / 差不多
    """

    return await judge_llm.invoke(pointwise_prompt)
```

**LLM-as-Judge 的已知偏差：**

| 偏差类型 | 描述 |
|---------|------|
| **位置偏差** | 倾向于给排在前面的回答更高分 |
| **冗长偏差** | 倾向于给更长的回答更高分 |
| **自我偏好** | GPT-4 作为 Judge 倾向于给 GPT-4 的输出更高分 |
| **格式偏差** | 倾向于给格式更好看的回答更高分 |

**缓解偏差：**
- 交换 A/B 位置做两次评估取平均
- 使用与被评估模型不同的 Judge 模型
- 提供明确的评分 Rubric

### 四、人工评估

| 方法 | 具体做法 | 优势 | 劣势 | 适用场景 |
|------|---------|------|------|----------|
| **专家评审** | 领域专家按预定标准打分 | 质量最高，能评估专业领域的细微差别 | 成本高，速度慢，难规模化 | 高风险场景（医疗、法律、金融） |
| **众包标注** | 通过 Scale AI、Toloka 等平台招募标注者 | 可规模化，成本相对可控 | 标注者质量参差不齐，需要质量控制 | 大规模偏好数据收集 |
| **用户反馈** | 收集真实用户的点赞/点踩/投诉 | 最真实的质量信号 | 反馈稀疏（大部分用户不反馈），有偏差 | 生产环境的持续监控 |

### 五、混合评估框架（推荐）

```python
class HybridEvaluator:
    """混合评估：自动指标 + LLM Judge + 人工抽检"""

    async def evaluate(self, test_set):
        results = []

        for example in test_set:
            prediction = await self.model.invoke(example.input)
            scores = {}

            # Layer 1: 自动指标（全量，毫秒级）
            scores["exact_match"] = exact_match(prediction, example.reference)
            scores["f1"] = token_f1(prediction, example.reference)

            # Layer 2: LLM-as-Judge（全量或采样，秒级）
            scores["llm_judge"] = await llm_as_judge(
                example.input, prediction, example.reference
            )

            # Layer 3: 标记需要人工审核的案例
            if scores["llm_judge"]["total"] < 3 or scores["f1"] < 0.5:
                scores["needs_human_review"] = True

            results.append(scores)

        # Layer 3 继续：人工审核低分和边界案例
        flagged = [r for r in results if r.get("needs_human_review")]
        # 送人工审核队列...

        return results
```

### 六、评估指标选择指南

| 任务类型 | 推荐指标 | 评估方式 |
|---------|---------|----------|
| 事实性问答 | 精确匹配/F1 | 自动 |
| 文本摘要 | ROUGE + LLM | 自动 + LLM Judge |
| 翻译 | BLEU + 人工 | 自动 + 人工 |
| 创意写作 | 人工 + LLM | LLM Judge + 人工 |
| 对话质量 | LLM 多维评分 | LLM Judge |
| 代码生成 | Pass@k | 自动（运行测试） |
| Agent 任务 | 任务完成率 | 自动 + 轨迹评估 |
| 安全性 | 拒绝率/攻击成功 | 自动 + 红队测试 |

### 七、从 LLM 评估到 Agent 评估——核心差异

LLM 评估的三大方法（自动指标、人工评估、LLM-as-Judge）仍然适用于 Agent，但 Agent 的评估需要根本性扩展。

**LLM 评估：** 输入 → [LLM] → 输出，评估输出的质量（准确、流畅、有用）

**Agent 评估：** 任务 → [推理] → [工具调用] → [观察] → [推理] → [工具调用] → ... → 结果

Agent 评估的六个维度：
1. **最终结果** — 任务是否完成？
2. **推理质量** — 每步推理是否合理？
3. **工具使用** — 工具选择是否正确？参数是否正确？
4. **效率** — 用了多少步？花了多少 token？
5. **错误恢复** — 遇到错误是否能恢复？
6. **安全性** — 是否有越权操作？

| 维度 | LLM 评估 | Agent 评估 |
|------|---------|------------|
| 评估对象 | 单次生成 | 多步决策轨迹 |
| 评估范围 | 输出文本质量 | 任务完成 + 过程 |
| 确定性 | 较高 | 低（多路径可行） |
| 关键指标 | 准确率、BLEU | 任务完成率、效率 |
| 工具使用 | 无 | 核心评估维度 |
| 安全性 | 输出安全 | 行为安全（操作） |
| 评估复杂度 | 低 | 高 |
| 基准测试 | MMLU、GSM8K | SWE-bench、GAIA |

ACL 2025 的 Agent 评估综述提出二维分类法：评估"什么能力"（推理、规划、工具使用等） x "用什么方法"（基准测试、人工评估、LLM Judge 等）。

### 八、Agent 评估的四层模型

| 层级 | 核心问题 | 指标 | 方法 | 示例 |
|------|---------|------|------|------|
| **Layer 1 — 结果评估（What）** | Agent 是否完成了任务？ | 任务完成率、答案准确率、部分完成度 | 自动化检查最终状态 | SWE-bench: 代码修改后测试是否通过 |
| **Layer 2 — 轨迹评估（How）** | Agent 的决策路径是否合理？ | 步骤合理性、是否有冗余步骤、是否走了弯路 | LLM-as-Judge 或人工评估 Trace | 10 步完成 vs 3 步完成，效率差异巨大 |
| **Layer 3 — 工具评估（With What）** | Agent 是否正确使用了工具？ | 工具选择准确率、参数正确率、调用次数 | 与最优工具使用序列对比 | 搜索 vs 计算——应该用计算器时却去搜索 |
| **Layer 4 — 鲁棒性评估（What If）** | Agent 面对异常情况如何表现？ | 错误恢复率、幻觉率、安全违规率 | 注入故障和对抗样本 | 工具返回错误时是否能换策略重试 |

### 九、轨迹评估（Trajectory Evaluation）

```python
class TrajectoryEvaluator:
    """评估 Agent 的完整执行轨迹"""

    async def evaluate_trajectory(self, task, trajectory):
        scores = {}

        # 1. 步骤级评估：每一步是否合理
        step_scores = []
        for i, step in enumerate(trajectory.steps):
            step_score = await self.evaluate_step(
                task=task,
                step=step,
                context=trajectory.steps[:i],  # 前序上下文
            )
            step_scores.append(step_score)
        scores["step_quality"] = np.mean(step_scores)

        # 2. 轨迹效率：是否有冗余步骤
        scores["efficiency"] = self.compute_efficiency(
            actual_steps=len(trajectory.steps),
            optimal_steps=self.get_optimal_length(task),
        )

        # 3. 目标达成度
        scores["goal_achieved"] = await self.check_goal(
            task=task,
            final_state=trajectory.final_state,
        )

        # 4. 错误恢复：遇到错误后的处理
        errors = [s for s in trajectory.steps if s.is_error]
        if errors:
            recovery_rate = sum(1 for e in errors if e.was_recovered) / len(errors)
            scores["error_recovery"] = recovery_rate

        return scores

    async def evaluate_step(self, task, step, context):
        """用 LLM 评估单步决策"""
        return await self.judge_llm.invoke(f"""
        任务：{task}
        已执行步骤：{context}
        当前步骤：{step}

        评估这一步是否合理（1-5分）：
        - 是否推进了任务目标？
        - 工具选择是否正确？
        - 参数是否合理？
        """)
```

### 十、主要 Agent 基准测试

| Agent 类型 | 基准测试 | 评估内容 |
|-----------|---------|----------|
| **代码 Agent** | SWE-bench | 修复真实 GitHub Issue（Resolved Rate） |
| | HumanEval | 生成函数代码（Pass@k） |
| | MBPP | Python 编程任务（Pass@k） |
| **Web Agent** | WebArena | 在真实网站完成复杂任务 |
| | Mind2Web | 跨网站的通用网页操作 |
| | VisualWebArena | 需要视觉理解的网页任务 |
| **通用推理 Agent** | ALFWorld | 文本版家庭环境中的任务执行 |
| | WebShop | 模拟电商购物任务 |
| | GAIA | 通用 AI 助手评估（需要多工具组合） |
| **工具使用 Agent** | ToolBench | 评估 API 工具的选择和使用 |
| | API-Bank | 评估 API 调用的正确性 |
| | TaskBench | 多工具组合任务 |

### 十一、生产环境 Agent 评估框架

```python
class ProductionAgentEvaluator:
    """生产环境中的 Agent 评估"""

    def __init__(self):
        # 评估指标体系
        #
        # | 类别 | 指标 | 含义 |
        # |------|------|------|
        # | **核心** | task_success_rate | 任务完成率 |
        # | | avg_steps | 平均步骤数 |
        # | | avg_latency | 平均延迟 |
        # | | avg_cost | 平均成本 |
        # | **质量** | trajectory_quality | 轨迹质量（LLM Judge） |
        # | | tool_accuracy | 工具使用准确率 |
        # | | hallucination_rate | 幻觉率 |
        # | **安全** | safety_violation_rate | 安全违规率 |
        # | | unauthorized_action_rate | 越权操作率 |
        pass

    async def run_eval_suite(self, agent, test_cases):
        results = []
        for case in test_cases:
            # 执行并记录完整轨迹
            trajectory = await agent.execute_with_trace(case.task)

            # 多维评估
            eval_result = {
                "task_success": self.check_success(trajectory, case.expected),
                "steps": len(trajectory.steps),
                "cost": trajectory.total_cost,
                "latency": trajectory.total_time,
                "trajectory_score": await self.judge_trajectory(trajectory),
                "tool_accuracy": self.check_tool_usage(trajectory),
                "safety": self.check_safety(trajectory),
            }
            results.append(eval_result)

        return self.aggregate(results)
```

## 常见误区 / 面试追问

1. **误区："BLEU/ROUGE 分数高就说明质量好"** — 这些指标只衡量表面词汇重叠，无法评估语义正确性、逻辑合理性和实用性。两个语义相同但措辞不同的回答可能得到很不同的 BLEU 分数。LLM 时代这些传统指标的参考价值有限。

2. **误区："LLM-as-Judge 完全可以替代人工"** — LLM Judge 有系统性偏差（冗长偏好、位置偏差、自我偏好），且在专业领域（医学、法律）的判断可能不可靠。生产中应该定期用人工评估校准 LLM Judge 的准确性。

3. **误区："Agent 评估只看最终结果就够了"** — 最终结果正确但过程不合理的 Agent 同样有问题——可能走了弯路浪费资源，可能碰巧得到正确结果但推理错误（不可靠），可能使用了不安全的操作。轨迹评估和结果评估同等重要。

4. **误区："用 LLM 基准测试就能评估 Agent"** — LLM 基准（如 MMLU）测试的是知识和推理能力，无法反映 Agent 的工具使用、规划和错误恢复能力。Agent 需要专用基准（如 SWE-bench、WebArena、GAIA）。

5. **追问："如何提高 LLM-as-Judge 的可靠性？"** — (1) 提供详细的评分 Rubric（标准）而非让 Judge 自由打分；(2) 交换位置做两次评估取平均（消除位置偏差）；(3) 用多个 Judge 模型投票；(4) 定期用人工标注校准。

6. **追问："评估数据集从哪里来？"** — 三个来源：(1) 从生产日志中采样真实问题；(2) 人工构造边界案例和对抗样本；(3) 使用公开基准（MMLU、GSM8K 等）。最佳实践是三者结合——公开基准评估通用能力，私有数据集评估业务场景。

7. **追问："如何评估 Agent 的效率？"** — 三个维度：(1) 步骤效率——完成任务用了多少步（vs 最优步数）；(2) 成本效率——消耗了多少 token/金钱；(3) 时间效率——端到端延迟。权衡是：更多步骤可能提升准确率但增加成本。

8. **追问："Agent 评估的最大难点是什么？"** — 非确定性。同一任务可能有多条正确路径，无法用固定的"标准答案"对比。解决方案：(1) 评估最终状态而非中间步骤；(2) 用 LLM Judge 评估轨迹的合理性；(3) 多次运行取统计指标。

## 参考资料

- [Understanding the 4 Main Approaches to LLM Evaluation (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/llm-evaluation-4-approaches)
- [LLM Evaluation Metrics: The Ultimate Guide (Confident AI)](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
- [LLM Evaluation: Benchmarks vs. Human Judgment (Medium)](https://medium.com/@lmpo/llm-evaluation-benchmarks-vs-human-judgment-f1cdd16098c0)
- [LLM Evaluation Metrics and Methods, Explained Simply (Evidently AI)](https://www.evidentlyai.com/llm-guide/llm-evaluation-metrics)
- [An Analysis of Automated, Human, and LLM-Based Approaches (arXiv)](https://arxiv.org/pdf/2406.03339)
- [Agent Evaluation vs Model Evaluation: What's the Difference (Maxim)](https://www.getmaxim.ai/articles/agent-evaluation-vs-model-evaluation-whats-the-difference-and-why-it-matters/)
- [Evaluation and Benchmarking of LLM Agents: A Survey (ACL 2025)](https://arxiv.org/html/2507.21504v1)
- [LLM Agent Evaluation: Assessing Tool Use, Task Completion (Confident AI)](https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide)
- [The Complete Guide to LLM & AI Agent Evaluation in 2026 (Adaline)](https://www.adaline.ai/blog/complete-guide-llm-ai-agent-evaluation-2026)
- [Understanding How AI Agent Trajectories Guide Agent Evaluation (Objectways)](https://objectways.com/blog/understanding-how-ai-agent-trajectories-guide-agent-evaluation/)
