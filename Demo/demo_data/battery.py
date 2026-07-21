"""
电池材料领域 Mock 数据生成器。

从 battery_demo.py 提取，提供结构化的假设、实验、结果和分析数据。
"""

from typing import List, Dict


def generate_hypotheses(iteration: int) -> List[Dict]:
    """生成电池电解质相关的科学假设。"""
    base = [
        {
            "id": f"H{iteration}_1",
            "statement": "石榴石型 LLZO 中 Al/Ga 共掺杂可以协同提升离子电导率和稳定性",
            "scientific_basis": "Al3+ 和 Ga3+ 的离子半径差异可以优化晶格参数，扩大 Li 离子传输通道，同时稳定立方相结构",
            "predicted_outcome": "室温离子电导率达到 1-3 mS/cm，电化学窗口>5V",
            "confidence": 0.82,
            "generation": iteration,
            "parent_ids": [],
            "status": "active",
        },
        {
            "id": f"H{iteration}_2",
            "statement": "在正极/电解质界面引入超薄 LiNbO3 缓冲层可以抑制副反应并降低界面阻抗",
            "scientific_basis": "LiNbO3 具有良好的锂离子导电性和化学稳定性，可以作为保护层阻止空间电荷层形成",
            "predicted_outcome": "界面阻抗降低 80%，循环稳定性提升至 500 圈以上",
            "confidence": 0.76,
            "generation": iteration,
            "parent_ids": [],
            "status": "active",
        },
        {
            "id": f"H{iteration}_3",
            "statement": "基于机器学习的主动学习策略可以高效筛选出新型超离子导体",
            "scientific_basis": "通过迭代训练代理模型并选择信息量最大的样本进行 DFT 计算，可以快速探索广阔的化学空间",
            "predicted_outcome": "在 1000 个候选物中发现 3-5 个离子电导率>10 mS/cm 的新材料",
            "confidence": 0.71,
            "generation": iteration,
            "parent_ids": [],
            "status": "active",
        },
    ]

    if iteration >= 2:
        base.append({
            "id": f"H{iteration}_4",
            "statement": "硫化物 - 氧化物复合电解质可以实现性能互补和界面优化",
            "scientific_basis": "硫化物具有高离子电导率但稳定性差，氧化物稳定性好但电导率低，复合后可能协同优化",
            "predicted_outcome": "综合离子电导率>2 mS/cm，同时对空气和锂金属稳定",
            "confidence": 0.68,
            "generation": iteration,
            "parent_ids": [f"H{iteration-1}_1", f"H{iteration-1}_2"],
            "status": "active",
        })

    return base


def plan_experiments(hypotheses: List[Dict], iteration: int) -> List[Dict]:
    """为假设设计计算实验方案。"""
    experiments = []
    for hyp in hypotheses:
        if "石榴石" in hyp["statement"] or "LLZO" in hyp["statement"]:
            exp = {
                "hypothesis_id": hyp["id"],
                "method": "DFT + MD",
                "tool_chain": ["VASP", "LAMMPS", "pymatgen"],
                "steps": [
                    "构建 Al/Ga 掺杂的 LLZO 晶体结构模型",
                    "使用 VASP 进行几何优化和电子结构计算",
                    "计算 Li 离子迁移路径和能垒 (NEB 方法)",
                    "使用 LAMMPS 进行 AIMD 模拟计算扩散系数",
                    "分析径向分布函数和均方位移",
                    "计算电化学稳定性和对锂润湿性",
                ],
                "code_specification": "使用 pymatgen 构建掺杂结构，VASP 进行静态计算，LAMMPS 进行高温 MD...",
                "success_criteria": "离子电导率计算值>0.5 mS/cm，迁移能垒<0.3 eV",
                "estimated_complexity": "high",
            }
        elif "界面" in hyp["statement"] or "LiNbO3" in hyp["statement"]:
            exp = {
                "hypothesis_id": hyp["id"],
                "method": "DFT 界面计算",
                "tool_chain": ["VASP", "pymatgen", "ASE"],
                "steps": [
                    "构建正极/缓冲层/电解质异质结模型",
                    "优化界面结构和结合能",
                    "计算界面处的元素互扩散势垒",
                    "分析界面电子结构和电荷转移",
                    "计算界面离子传输阻力",
                    "评估电化学稳定性窗口",
                ],
                "code_specification": "构建 slab 模型，优化界面结构，计算 PDOS 分析成键...",
                "success_criteria": "界面结合能<-0.5 J/m^2，界面阻抗<50 Ohm*cm^2",
                "estimated_complexity": "high",
            }
        else:
            exp = {
                "hypothesis_id": hyp["id"],
                "method": "ML_potential",
                "tool_chain": ["scikit-learn", "pymatgen", "matminer"],
                "steps": [
                    "从 Materials Project 下载 10000+ 含锂化合物数据",
                    "提取结构描述符和成分特征",
                    "训练 GBDT 模型预测离子电导率",
                    "使用主动学习策略选择最有价值的样本",
                    "对 Top 50 候选物进行 DFT 验证",
                    "分析构效关系并提炼设计规则",
                ],
                "code_specification": "使用 matminer 提取特征，scikit-learn 训练模型，交叉验证...",
                "success_criteria": "模型 R^2>0.8，发现至少 3 个电导率>1 mS/cm 的新材料",
                "estimated_complexity": "medium",
            }
        experiments.append(exp)
    return experiments


def execute_experiments(experiments: List[Dict]) -> List[Dict]:
    """模拟执行电池材料计算实验。"""
    results = []
    for exp in experiments:
        if "LLZO" in str(exp) or "石榴石" in str(exp):
            result = {
                "hypothesis_id": exp["hypothesis_id"],
                "plan_summary": f"DFT+MD: {exp['steps'][0]}",
                "success": True,
                "final_code": "from pymatgen.core import Structure\n# LLZO doping calculation...",
                "execution_output": "计算完成！离子电导率符合预期，Al/Ga 共掺杂效果显著",
                "error_log": "",
                "key_findings": [
                    "Al/Ga 共掺杂确实稳定了立方相结构",
                    "Li 离子迁移通道扩大了 15%",
                    "计算得到的离子电导率为 1.8 mS/cm (300K)",
                    "电化学窗口达到 5.2V",
                ],
                "numerical_results": {
                    "离子电导率": "1.8 mS/cm",
                    "迁移能垒": "0.25 eV",
                    "晶格膨胀": "2.3%",
                    "形成能": "-3.45 eV/atom",
                },
            }
        elif "界面" in str(exp) or "LiNbO3" in str(exp):
            result = {
                "hypothesis_id": exp["hypothesis_id"],
                "plan_summary": f"DFT: {exp['steps'][0]}",
                "success": True,
                "final_code": "from ase.build import surface\n# Interface calculation...",
                "execution_output": "界面计算成功，缓冲层效果显著",
                "error_log": "",
                "key_findings": [
                    "LiNbO3 与两侧材料都有良好的晶格匹配",
                    "界面处形成了稳定的化学键",
                    "Li 离子跨界面迁移能垒仅为 0.18 eV",
                    "有效抑制了空间电荷层形成",
                ],
                "numerical_results": {
                    "界面结合能": "-0.72 J/m^2",
                    "界面阻抗": "28 Ohm*cm^2",
                    "层厚优化": "5-8 nm",
                    "稳定性提升": "3.2 倍",
                },
            }
        else:
            result = {
                "hypothesis_id": exp["hypothesis_id"],
                "plan_summary": f"ML: {exp['steps'][0]}",
                "success": True,
                "final_code": "from sklearn.ensemble import GradientBoostingRegressor\n# ML screening...",
                "execution_output": "ML 筛选完成，发现多个有前景的材料",
                "error_log": "",
                "key_findings": [
                    "主动学习策略比随机筛选效率高 8 倍",
                    "发现了 4 个潜在超离子导体候选物",
                    "识别出关键的描述符：四面体空隙大小、Li 含量",
                    "建立了可靠的结构 - 性能关系",
                ],
                "numerical_results": {
                    "模型 R^2": "0.83",
                    "筛选效率": "8x 提升",
                    "候选物数量": "4 个",
                    "最高预测电导率": "12.5 mS/cm",
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
            boost = 0.12
            verdict = "SUPPORTS"
            reasoning = f"实验数据强力支持预测，{list(result['numerical_results'].keys())[0]}达到预期"
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
            "statistical_summary": "数值结果与预测高度吻合" if verdict == "SUPPORTS" else "数值偏差较大",
            "explanation": f"{'实验结果强力支持该假设，所有关键指标均达到成功标准。' if verdict == 'SUPPORTS' else '实验结果不支持该假设。'}",
        })

        if verdict == "SUPPORTS":
            updated_ids.append(hyp["id"])

    new_hypotheses = []
    if iteration >= 2 and len(hypotheses) >= 2:
        new_hypotheses.append({
            "id": f"H{iteration}_new",
            "statement": "多元协同策略：Al/Ga 掺杂 + 界面缓冲层 + 机器学习优化的复合电解质体系",
            "confidence": 0.65,
            "generation": iteration,
            "status": "active",
            "parent_ids": [h["id"] for h in hypotheses[:2]],
        })

    return {
        "evaluations": evaluations,
        "analyses": analyses,
        "updated_ids": updated_ids,
        "eliminated_ids": [],
        "new_hypotheses": new_hypotheses,
        "evolution_reasoning": f"第 {iteration} 轮迭代中，掺杂策略和界面工程均得到实验验证。"
                               f"机器学习筛选高效发现候选材料。"
                               f"{'提出多元协同的综合策略假设。' if iteration >= 2 else ''}",
        "converged": iteration >= 3,
    }
