# SART 投资人通话演讲稿

> **场景**：与 founder-scout 的 30 分钟视频通话。对方不是传统 VC，在寻找有潜力的 builder 和潜在创始人。已表示可以说中文。
>
> **核心策略**：不做 presentation，做 conversation。以下是结构化的谈话要点，不是逐字稿——准备好被打断和追问。

---

## 开场（2–3 分钟）

**直接从问题切入，不从学术背景开始。**

> 谢谢你的时间。我先用一句话说清楚我在做什么：
>
> 现在企业花了很多资源去 fine-tune 大语言模型，但这些模型上线之后会在真实场景里静默地失败——benchmark 上 95 分，到了真实用户那里可能只有 57 分。这个 gap 是 20 到 40 个百分点，而且是 invisible 的，你直到上线才会发现。
>
> 我发现了这个问题的根本原因：模型在训练的时候学到的不是真正的推理，而是数据里的 shortcut——表面的 pattern。
>
> **而且我发现，现在市场上没有任何工具能在训练阶段检测和修复这个问题。**
>
> 所以我做了 SART，一个 training-time 的诊断和修复框架。

---

## 第一部分：我做了什么（5 分钟）

**重点展示执行力，不是技术细节。**

> SART 做两件事：第一，用 gradient signal 自动检测哪些训练样本在教模型走捷径；第二，在训练过程中实时修复——要么降低这些样本的权重，要么用 gradient surgery 把有害的梯度方向投影掉。
>
> 关键是：**不需要事先知道 shortcut 是什么类型**。Gradient 的信号会自动把它们暴露出来。
>
> 这意味着它可以 plug into 任何现有的 fine-tuning pipeline，作为一个 callback，不需要重写训练流程。

**如果对方追问技术细节：**

> 核心是一个叫 ShortcutScore 的 per-sample metric。它衡量每个训练样本的 gradient 和 validation set 的 gradient 之间的 alignment。如果一个样本的 gradient 方向和 validation 的方向不一致，而且 loss 集中在 answer token 上，那这个样本大概率在教 shortcut。
>
> 这个 score 驱动两个 correction：sample reweighting 和 gradient surgery。两个可以单独用，也可以叠加——叠加的效果是 super-additive 的，比各自效果之和还要多 26.7 个百分点。

**用数字说话：**

> 我对比了 12 个现有方法——包括 Group DRO、IRM、JTT、Focal Loss 这些——在 5 个数据集上做了完整的 benchmark。结果是：
>
> - 比 standard fine-tuning 高 **+16 个百分点的 accuracy**，**+38 个百分点的 robustness**
> - 在最难的 Financial Analysis 数据集上，打赢了 Group DRO 4.2 个百分点
>
> 这不是 notebook 级别的实验。整个 codebase 是 production-grade 的：一条命令跑通全流程，MLflow tracking，Optuna Bayesian search，双 scale profile（本地调试和服务器训练），SLURM 支持。13 个方法共享一个 common interface，新方法只需要实现一个函数就能 plug in。

---

## 第二部分：为什么是现在（3 分钟）

> 这个问题为什么现在特别 urgent？四个原因：
>
> 第一，**regulated 行业正在大规模采用 LLM**——金融、医疗、法律。这些领域的模型不能 silent fail，错误是有法律后果的。
>
> 第二，**fine-tuning 会放大 shortcut 问题**。因为 domain 数据通常比较小，pattern 更集中，模型更容易走捷径。而 fine-tuning 恰恰是企业最常用的 LLM 定制方式。
>
> 第三，**监管压力是真实的**。EU AI Act 2026 年开始执行，FDA 的 AI/ML guidance 越来越严。高风险 AI 系统需要提供 demonstrable robustness 的证据。
>
> 第四，也是最关键的：**现在没有任何 production 级别的工具做这件事**。现有的方法要么需要你提前知道 shortcut 是什么（data augmentation），要么需要训练两遍（JTT），要么只能做 group-level 的处理（DRO）。没有一个能在 training time 自动地、per-sample 地检测和修复。

---

## 第三部分：商业化方向（5 分钟）

**这是对方最想听的部分。**

> 我看到三个产品方向，按优先级排：

### 方向一：Training-Time Diagnostic API（SaaS）

> 最直接的 wedge 是一个诊断 API。调用方式就是 `sart.diagnose(model, train_data, val_data)`，输出每个样本的 shortcut score 报告。
>
> Go-to-market 的想法是：free tier 做诊断，付费做 correction。类似 Weights & Biases 的模式——免费 hook 住用户，用 value 升级。
>
> 第一批客户画像：Series B 以上的创业公司里的 ML platform team，他们在做 domain fine-tuning，已经踩过 production failure 的坑。Budget holder 是 VP Engineering 或 Head of ML。

### 方向二：Robustness-Certified Fine-Tuning（平台合作）

> 和 fine-tuning provider 合作——Together AI、Anyscale、Fireworks 这类公司。Value proposition 是："你 fine-tune 出来的模型，我们保证 X% 的 robustness"。

### 方向三：Compliance Tooling（合规市场）

> 为 regulated 行业做 audit-ready 的 robustness report。Entry point 是 fintech 公司的 model risk management（SR 11-7 compliance）。

### 90 天 Go-to-Market

> 如果资源到位，我的 90 天计划是：
>
> - 前 4 周：做 20 个 customer discovery interview，和 ML platform lead 聊痛点和 willingness to pay
> - 第 5–8 周：把 ShortcutScore 包装成一个 diagnostic API 的 MVP
> - 第 9–12 周：找 3 个 design partner 在他们的 pipeline 上跑 pilot
>
> 这样 90 天结束的时候我有：validated pain points、working product、real usage data。

---

## 第四部分：我需要什么（3 分钟）

**具体的 ask，不是泛泛地说"我需要融资"。**

> 我很清楚现在处在什么阶段——技术已经 validated，但商业化还在最早期。我需要四样东西：
>
> 第一，**co-founder**。我需要一个有 developer tools 或 MLOps 领域 go-to-market 经验的合伙人。我的强项是算法和工程，但 enterprise sales 和 pricing 不是我的专长。
>
> 第二，**3 到 6 个月的 runway**。够完成 customer discovery 和 build diagnostic MVP。
>
> 第三，**introductions**。如果你认识正在做 LLM fine-tuning 的团队，我想和他们聊——不是卖东西，而是 validate 痛点。
>
> 第四，**mentorship**。公司怎么注册、怎么定价、早期 enterprise sales 怎么做——这些我需要有经验的人指导。

---

## 收尾（3–5 分钟）

**准备好反问，展示你在认真思考这件事。**

> 我想反过来问你几个问题：
>
> 1. 你们见过的类似 AI infra / tooling 的创始人，最容易踩的坑是什么？
> 2. 从你们的角度看，像我这样从研究驱动的项目起步，什么时间点适合正式开始融资？
> 3. 你们 portfolio 里有没有做 MLOps 或 fine-tuning 相关的公司？可能有 synergy 或者可以介绍给我做 customer discovery。

---

## 预备问答

### "如果 OpenAI 自己解决了 robustness？"

> 大模型厂商的 incentive 是通用能力，不是 fine-tuning 的质量保证。这是一个 tooling 层的问题——就像 Weights & Biases 没有被 PyTorch 取代一样，training diagnostics 是一个独立的价值层。而且 shortcut detection 在学术界仍然是一个 open research problem，没有任何一个 provider 把它做成产品了。

### "Gradient computation 在大模型上跑得动吗？"

> 这是主要的技术挑战。我已经实现了 gradient sketching 来做近似计算，在实验的 scale 上效果没有损失。下一步是在 7B 参数的模型上做 benchmark。如果全量 gradient 在超大 scale 上确实不可行，可以退到 adapter 层做——LoRA fine-tuning 的场景下，gradient 的维度会小很多。

### "你一个人能做公司吗？"

> 我不打算一个人做。但我想强调的是：一个人能做到现在这个程度——完整的算法设计、13 个 baseline 对比、production-grade 的工程框架、NeurIPS submission——这本身就是执行力的证明。代码库是模块化的、有完整文档的，设计上就是为团队协作准备的。我需要的 co-founder 是在 go-to-market 上能补位的人。

### "为什么不先去大厂积累几年经验？"

> 时间窗口。EU AI Act 2026 年执行，fine-tuning 市场正在爆发，目前没有竞品。如果等两三年，要么这个机会被别人做了，要么大公司内部解决了。现在是 first-mover 的最佳时机。

### "你的 moat 是什么？别人不能重新实现吗？"

> 算法本身是可以重新实现的，我不否认。但 moat 不在算法的 secrecy，而在三个地方：第一，benchmark 的深度——13 个方法、5 个数据集的完整对比，这是行业里最全的，这决定了 credibility。第二，执行速度——从 idea 到 production-grade implementation 到 NeurIPS 的速度。第三，也是最重要的，是 deployment 之后的数据飞轮——每次在客户的数据上跑，都会发现新的 shortcut 类型，detection 就变得更准。这个飞轮一旦转起来，后来者很难追。

---

## 通话注意事项

1. **不要做 presentation，做 conversation**。对方说的是"call"不是"pitch"，准备好被打断
2. **语言策略**：开场用英文展示专业度，如果对方切中文就自然切过去。技术术语保持英文
3. **诚实但不自我削弱**。"我还没做 customer discovery"可以说，但紧跟"我的 90 天计划是..."
4. **时间控制**：你说话的时间不超过 60%，留足够空间让对方追问
5. **Single-author 不是弱点，是信号**。主动提："一个人做到这个程度说明执行力，但我清楚地知道我需要 co-founder"
