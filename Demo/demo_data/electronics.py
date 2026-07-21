"""
电子信息与通信工程领域 Mock 数据生成器。

从 electronics_demo.py 提取，提供结构化的假设、实验、结果和分析数据。
"""

from typing import List, Dict


def generate_hypotheses(iteration: int) -> List[Dict]:
    """生成电子信息与通信工程相关的科学假设。"""
    base = [
        {
            "id": f"H{iteration}_1",
            "statement": "基于深度强化学习的自适应调制编码策略可以动态优化频谱效率",
            "scientific_basis": "DRL 可以根据信道状态信息（CSI）实时选择最优的 MCS（调制编码方案），相比固定策略更能适应快变信道",
            "predicted_outcome": "频谱效率提升 35-50%，误码率保持在 10^-3 以下",
            "confidence": 0.79,
            "generation": iteration,
            "parent_ids": [],
            "status": "active",
        },
        {
            "id": f"H{iteration}_2",
            "statement": "Transformer 架构的信道估计网络可以显著降低导频开销并提高估计精度",
            "scientific_basis": "Transformer 的自注意力机制可以捕捉长距离依赖关系，从少量导频中恢复完整的信道响应",
            "predicted_outcome": "导频开销减少 60%，归一化均方误差（NMSE）<-35dB",
            "confidence": 0.74,
            "generation": iteration,
            "parent_ids": [],
            "status": "active",
        },
        {
            "id": f"H{iteration}_3",
            "statement": "轻量化 CNN-LSTM 混合模型可以实现低时延的信号检测与分类",
            "scientific_basis": "CNN 提取频域特征，LSTM 捕捉时域相关性，混合架构在精度和速度间取得平衡",
            "predicted_outcome": "检测准确率>98%，单帧处理时延<0.5ms，满足 URLLC 要求",
            "confidence": 0.71,
            "generation": iteration,
            "parent_ids": [],
            "status": "active",
        },
    ]

    if iteration >= 2:
        base.append({
            "id": f"H{iteration}_4",
            "statement": "多天线波束赋形的元学习优化可以快速适应新的传播环境",
            "scientific_basis": "元学习通过 meta-training 学习如何快速学习，只需少量样本就能调整波束赋形权重",
            "predicted_outcome": "在新环境中 5 次迭代内收敛，波束增益提升 8-12dB",
            "confidence": 0.66,
            "generation": iteration,
            "parent_ids": [f"H{iteration-1}_1"],
            "status": "active",
        })

    if iteration >= 3:
        base.append({
            "id": f"H{iteration}_5",
            "statement": "联邦学习框架可以在保护隐私的前提下协同优化多个基站的资源分配",
            "scientific_basis": "联邦学习只交换模型梯度而非原始数据，既保护用户隐私又实现全局优化",
            "predicted_outcome": "小区边缘用户吞吐量提升 40-70%，隐私泄露风险降低 90%",
            "confidence": 0.63,
            "generation": iteration,
            "parent_ids": [f"H{iteration-1}_1", f"H{iteration-1}_2"],
            "status": "active",
        })

    return base


def plan_experiments(hypotheses: List[Dict], iteration: int) -> List[Dict]:
    """为假设设计仿真实验方案。"""
    experiments = []
    for hyp in hypotheses:
        if "调制编码" in hyp["statement"] or "DRL" in hyp["statement"]:
            exp = {
                "hypothesis_id": hyp["id"],
                "method": "深度强化学习仿真",
                "tool_chain": ["PyTorch", "NumPy", "MATLAB"],
                "steps": [
                    "构建 5G NR 链路级仿真平台",
                    "实现 DRL 代理（PPO 算法）",
                    "定义状态空间（CQI, SNR, Doppler）",
                    "定义动作空间（MCS 索引 0-27）",
                    "设计奖励函数（吞吐量 - BLER 惩罚）",
                    "训练策略网络（10000 episodes）",
                    "测试集评估并与基准对比",
                ],
                "code_specification": "使用 PyTorch 构建 Actor-Critic 网络，输入为 CSI 反馈，输出为 MCS 选择...",
                "success_criteria": "平均频谱效率>4.5 bps/Hz，BLER<10%",
                "estimated_complexity": "high",
            }
        elif "Transformer" in hyp["statement"] or "信道估计" in hyp["statement"]:
            exp = {
                "hypothesis_id": hyp["id"],
                "method": "Transformer 网络训练",
                "tool_chain": ["PyTorch", "TensorFlow"],
                "steps": [
                    "生成 3GPP CDL 信道数据集（10 万样本）",
                    "构建 Transformer Encoder 架构",
                    "实现位置编码和多头注意力",
                    "训练模型预测完整信道矩阵",
                    "在不同 SNR 下测试 NMSE 性能",
                    "与传统 LS/MMSE 估计对比",
                ],
                "code_specification": "4 层 Encoder，8 头注意力，隐藏维度 256，使用 AdamW 优化器...",
                "success_criteria": "NMSE<-35dB@SNR=20dB，推理时间<1ms",
                "estimated_complexity": "high",
            }
        elif "CNN-LSTM" in hyp["statement"]:
            exp = {
                "hypothesis_id": hyp["id"],
                "method": "混合神经网络实现",
                "tool_chain": ["PyTorch", "CUDA"],
                "steps": [
                    "构建 CNN-LSTM 混合模型",
                    "CNN 分支：3 层 1D 卷积提取频域特征",
                    "LSTM 分支：2 层捕捉时域依赖",
                    "特征融合 + 全连接分类",
                    "在 RadioML 数据集上训练",
                    "在 Jetson Nano 上部署测试时延",
                ],
                "code_specification": "输入 IQ 两路信号，CNN 核大小 3x3，LSTM 隐藏单元 128...",
                "success_criteria": "准确率>98%，单帧时延<0.5ms",
                "estimated_complexity": "medium",
            }
        elif "波束赋形" in hyp["statement"] or "元学习" in hyp["statement"]:
            exp = {
                "hypothesis_id": hyp["id"],
                "method": "元学习优化仿真",
                "tool_chain": ["PyTorch", "MAML 库"],
                "steps": [
                    "构建大规模 MIMO 系统模型",
                    "实现 MAML（Model-Agnostic Meta-Learning）算法",
                    "Meta-training 在多种场景下",
                    "Meta-testing 在新场景快速适应",
                    "对比传统 ZF/MRT 波束赋形",
                    "分析收敛速度和波束增益",
                ],
                "code_specification": "内外循环优化，inner_lr=0.01，outer_lr=0.001，K=5 shots...",
                "success_criteria": "新场景 5 次迭代收敛，增益>8dB",
                "estimated_complexity": "high",
            }
        else:
            exp = {
                "hypothesis_id": hyp["id"],
                "method": "联邦学习分布式训练",
                "tool_chain": ["PyTorch", "Flower 框架"],
                "steps": [
                    "搭建联邦学习仿真环境（10 个基站）",
                    "实现 FedAvg 算法",
                    "各基站本地训练资源分配模型",
                    "聚合全局模型（加权平均）",
                    "评估系统级性能提升",
                    "分析隐私保护效果",
                ],
                "code_specification": "每轮选择 3 个基站参与，本地 epoch=5，batch_size=64...",
                "success_criteria": "边缘用户吞吐量提升>40%，收敛轮次<50",
                "estimated_complexity": "medium",
            }
        experiments.append(exp)
    return experiments


def execute_experiments(experiments: List[Dict]) -> List[Dict]:
    """模拟执行仿真实验。"""
    results = []
    for exp in experiments:
        if "DRL" in exp["method"] or "强化学习" in exp["method"]:
            result = {
                "hypothesis_id": exp["hypothesis_id"],
                "plan_summary": "DRL: 构建 5G NR 链路级仿真平台",
                "success": True,
                "final_code": "import torch\nimport torch.nn as nn\n# PPO Agent for MCS selection...",
                "execution_output": "DRL 训练完成，策略性能超越基准",
                "error_log": "",
                "key_findings": [
                    "DRL 策略成功学会根据 CQI 选择 MCS",
                    "在快衰落信道下表现优于固定策略",
                    "探索 - 利用平衡参数 epsilon 至关重要",
                    "训练稳定后 BLER 维持在目标值附近",
                ],
                "numerical_results": {
                    "平均频谱效率": "4.8 bps/Hz",
                    "频谱效率提升": "+42%",
                    "平均 BLER": "8.5%",
                    "收敛轮次": "约 6000 episodes",
                },
            }
        elif "Transformer" in exp["method"]:
            result = {
                "hypothesis_id": exp["hypothesis_id"],
                "plan_summary": "Transformer: 构建 Encoder 架构",
                "success": True,
                "final_code": "import torch\nfrom torch import nn\n# Transformer Channel Estimator...",
                "execution_output": "Transformer 训练成功，性能优于 MMSE",
                "error_log": "",
                "key_findings": [
                    "Transformer 成功从稀疏导频恢复信道",
                    "自注意力机制捕捉到频率选择性衰落",
                    "在低信噪比下仍保持良好性能",
                    "模型大小适中，可实时推理",
                ],
                "numerical_results": {
                    "NMSE@20dB": "-37.2 dB",
                    "导频开销减少": "65%",
                    "推理时间": "0.8 ms",
                    "参数量": "1.2M",
                },
            }
        elif "混合神经网络" in exp["method"] or "CNN-LSTM" in exp["method"]:
            result = {
                "hypothesis_id": exp["hypothesis_id"],
                "plan_summary": "CNN-LSTM: 构建混合模型",
                "success": True,
                "final_code": "import torch\nimport torch.nn as nn\n# CNN-LSTM Hybrid Detector...",
                "execution_output": "混合模型验证成功，满足 URLLC 要求",
                "error_log": "",
                "key_findings": [
                    "混合模型在 RadioML 上达到 SOTA",
                    "CNN 有效提取频域特征",
                    "LSTM 成功建模时序依赖",
                    "在嵌入式设备上实时运行",
                ],
                "numerical_results": {
                    "检测准确率": "98.7%",
                    "单帧时延": "0.42 ms",
                    "模型大小": "3.5 MB",
                    "功耗": "1.2 W",
                },
            }
        elif "元学习" in exp["method"]:
            result = {
                "hypothesis_id": exp["hypothesis_id"],
                "plan_summary": "元学习：实现 MAML 算法",
                "success": True,
                "final_code": "import torch\n# MAML for beamforming...",
                "execution_output": "元学习验证成功，快速适应新环境",
                "error_log": "",
                "key_findings": [
                    "元学习模型快速适应新场景",
                    "5 次梯度更新即达到满意性能",
                    "波束指向准确，旁瓣抑制良好",
                    "对天线阵列误差鲁棒",
                ],
                "numerical_results": {
                    "收敛迭代": "5 次",
                    "波束增益": "10.5 dB",
                    "旁瓣电平": "-18 dB",
                    "适应时间": "<10 ms",
                },
            }
        else:
            result = {
                "hypothesis_id": exp["hypothesis_id"],
                "plan_summary": "联邦学习：搭建分布式环境",
                "success": True,
                "final_code": "import torch\nfrom flower import fl\n# Federated learning...",
                "execution_output": "联邦学习验证成功，系统级性能提升",
                "error_log": "",
                "key_findings": [
                    "联邦学习成功协同优化多基站",
                    "无需共享原始数据即可提升性能",
                    "通信开销可接受（仅传输梯度）",
                    "隐私保护效果显著",
                ],
                "numerical_results": {
                    "边缘用户吞吐提升": "+58%",
                    "收敛轮次": "42 轮",
                    "通信开销": "2.3 MB/轮",
                    "隐私风险降低": "92%",
                },
            }
        results.append(result)
    return results


def analyze_and_evolve(hypotheses: List[Dict], results: List[Dict], iteration: int) -> Dict:
    """分析实验结果并进化假设。"""
    evaluations = []
    analyses = []
    updated_ids = []

    for hyp, result in zip(hypotheses, results):
        if result["success"]:
            boost = 0.13
            verdict = "SUPPORTS"
            reasoning = f"实验数据强力支持预测，{list(result['numerical_results'].keys())[0]}达到预期指标"
        else:
            boost = -0.15
            verdict = "REFUTES"
            reasoning = "实验结果与预测不符"

        evaluations.append({
            "hypothesis_id": hyp["id"],
            "verdict": verdict,
            "confidence_before": hyp["confidence"],
            "confidence_after": min(0.95, hyp["confidence"] + boost),
            "reasoning": reasoning,
        })

        analyses.append({
            "hypothesis_id": hyp["id"],
            "supports_hypothesis": verdict == "SUPPORTS",
            "confidence": min(0.95, hyp["confidence"] + boost),
            "key_findings": result.get("key_findings", [])[:3],
            "statistical_summary": "实验指标全面达到成功标准" if verdict == "SUPPORTS" else "关键指标未达标",
            "explanation": f"{'实验结果强力支持该假设，性能指标超越基准方案。' if verdict == 'SUPPORTS' else '实验结果不支持该假设。'}",
        })

        if verdict == "SUPPORTS":
            updated_ids.append(hyp["id"])

    new_hypotheses = []
    if iteration >= 3 and len(hypotheses) >= 3:
        new_hypotheses.append({
            "id": f"H{iteration}_new",
            "statement": "分层智能架构：物理层 Transformer+MAC 层 DRL+网络层联邦学习的跨层联合优化",
            "confidence": 0.62,
            "generation": iteration,
            "status": "active",
            "parent_ids": [h["id"] for h in hypotheses[:3]],
        })

    return {
        "evaluations": evaluations,
        "analyses": analyses,
        "updated_ids": updated_ids,
        "eliminated_ids": [],
        "new_hypotheses": new_hypotheses,
        "evolution_reasoning": f"第 {iteration} 轮迭代中，所有假设均获得实验支持。"
                               f"DRL 自适应策略和 Transformer 信道估计效果突出。"
                               f"{'提出跨层联合优化的综合假设。' if iteration >= 3 else ''}",
        "converged": iteration >= 4,
    }
