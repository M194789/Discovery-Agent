"""
催化科学领域 Mock 数据生成器。

从 demo.py 提取并扩展，提供结构化的假设、实验、结果和分析数据。
"""

from typing import List, Dict


def generate_hypotheses(iteration: int) -> List[Dict]:
    """生成催化科学领域的假设。"""
    base = [
        {
            "id": f"H{iteration}_1",
            "statement": "过渡金属掺杂可以降低 CO2 还原反应的能垒",
            "scientific_basis": "基于 d 带中心理论，过渡金属的 d 轨道电子结构可以调控吸附能",
            "predicted_outcome": "掺杂后的催化剂在较低过电位下实现高 CO2 转化效率",
            "confidence": 0.85,
            "generation": iteration,
            "parent_ids": [],
            "status": "active",
        },
        {
            "id": f"H{iteration}_2",
            "statement": "表面缺陷工程可以增加活性位点密度",
            "scientific_basis": "氧空位等缺陷可以作为 CO2 分子的吸附和活化中心",
            "predicted_outcome": "富含缺陷的表面展现出更高的电流密度和选择性",
            "confidence": 0.78,
            "generation": iteration,
            "parent_ids": [],
            "status": "active",
        },
        {
            "id": f"H{iteration}_3",
            "statement": "合金化可以协同优化电子结构和稳定性",
            "scientific_basis": "双金属效应可以同时调节活性和抗中毒能力",
            "predicted_outcome": "合金催化剂在长时间运行中保持高活性",
            "confidence": 0.72,
            "generation": iteration,
            "parent_ids": [],
            "status": "active",
        },
    ]

    if iteration >= 2:
        base.append({
            "id": f"H{iteration}_4",
            "statement": "单原子催化剂可以最大化原子利用效率并提供均一活性位点",
            "scientific_basis": "单原子分散的金属位点具有独特的电子结构和最高的原子效率",
            "predicted_outcome": "单原子催化剂在低金属负载下实现与纳米颗粒相当的催化活性",
            "confidence": 0.68,
            "generation": iteration,
            "parent_ids": [f"H{iteration-1}_1", f"H{iteration-1}_3"],
            "status": "active",
        })

    return base


def plan_experiments(hypotheses: List[Dict], iteration: int) -> List[Dict]:
    """为催化假设设计计算实验。"""
    experiments = []
    for hyp in hypotheses:
        if "掺杂" in hyp["statement"] or "过渡金属" in hyp["statement"]:
            exp = {
                "hypothesis_id": hyp["id"],
                "method": "DFT",
                "tool_chain": ["pymatgen", "ase"],
                "steps": [
                    "构建过渡金属掺杂的表面模型",
                    "计算 CO2 吸附能",
                    "计算反应路径和能垒",
                    "分析电子结构（d 带中心）",
                ],
                "code_specification": "使用 pymatgen 构建表面 slab 模型，掺杂不同过渡金属...",
                "success_criteria": "找到吸附能在 -0.5 到 -1.5 eV 之间的催化剂",
                "estimated_complexity": "high",
            }
        elif "缺陷" in hyp["statement"]:
            exp = {
                "hypothesis_id": hyp["id"],
                "method": "DFT + MD",
                "tool_chain": ["VASP", "pymatgen"],
                "steps": [
                    "构建含氧空位的表面模型",
                    "计算缺陷形成能",
                    "计算 CO2 在缺陷位点的吸附构型",
                    "分析缺陷浓度对活性的影响",
                ],
                "code_specification": "使用 VASP 进行缺陷计算，pymatgen 分析电荷密度...",
                "success_criteria": "缺陷位点吸附能比完美表面低 0.3 eV 以上",
                "estimated_complexity": "high",
            }
        elif "合金" in hyp["statement"]:
            exp = {
                "hypothesis_id": hyp["id"],
                "method": "DFT",
                "tool_chain": ["pymatgen", "ase"],
                "steps": [
                    "构建双金属合金表面模型",
                    "计算合金稳定性和混合焓",
                    "计算 CO2 还原反应路径",
                    "对比纯金属和合金的催化性能",
                ],
                "code_specification": "使用 ase 构建合金 slab，优化表面结构...",
                "success_criteria": "合金催化剂能垒低于纯金属 0.2 eV 以上",
                "estimated_complexity": "medium",
            }
        else:
            exp = {
                "hypothesis_id": hyp["id"],
                "method": "ML_potential",
                "tool_chain": ["scikit-learn", "pymatgen", "matminer"],
                "steps": [
                    "收集催化剂数据库中的性能数据",
                    "提取结构和电子特征描述符",
                    "训练机器学习模型预测催化活性",
                    "筛选最优候选催化剂",
                ],
                "code_specification": "使用 matminer 提取特征，scikit-learn 训练预测模型...",
                "success_criteria": "模型 R^2>0.8，筛选出 3 个以上候选材料",
                "estimated_complexity": "medium",
            }
        experiments.append(exp)
    return experiments


def execute_experiments(experiments: List[Dict]) -> List[Dict]:
    """模拟执行催化实验。"""
    results = []
    for exp in experiments:
        if "掺杂" in str(exp.get("steps", [])) or "过渡金属" in str(exp.get("steps", [])):
            result = {
                "hypothesis_id": exp["hypothesis_id"],
                "plan_summary": f"DFT: {exp['steps'][0]}",
                "success": True,
                "final_code": "from ase import Atoms\nfrom pymatgen.core import Structure\n# DFT calculation...",
                "execution_output": "吸附能计算结果：-1.23 eV (Cu 掺杂)\n能垒：0.45 eV\n符合预期!",
                "error_log": "",
                "key_findings": [
                    "Cu 掺杂降低了 CO2 还原能垒 0.35 eV",
                    "d 带中心从 -2.1 eV 移至 -1.8 eV",
                    "CO2 吸附能为 -1.23 eV，在最优区间内",
                    "Fe 和 Co 掺杂效果次之",
                ],
                "numerical_results": {
                    "吸附能 (Cu)": "-1.23 eV",
                    "能垒降低": "0.35 eV",
                    "d 带中心偏移": "0.3 eV",
                    "法拉第效率": "78%",
                },
            }
        elif "缺陷" in str(exp.get("steps", [])):
            result = {
                "hypothesis_id": exp["hypothesis_id"],
                "plan_summary": f"DFT+MD: {exp['steps'][0]}",
                "success": True,
                "final_code": "from pymatgen.core import Structure\n# Defect calculation...",
                "execution_output": "氧空位形成能：2.1 eV\n缺陷位点吸附能：-1.45 eV\n活性提升显著",
                "error_log": "",
                "key_findings": [
                    "氧空位形成能为 2.1 eV，热力学可行",
                    "缺陷位点 CO2 吸附能比完美表面低 0.42 eV",
                    "缺陷浓度 5% 时活性最优",
                    "过高缺陷浓度导致结构不稳定",
                ],
                "numerical_results": {
                    "缺陷形成能": "2.1 eV",
                    "吸附能改善": "-0.42 eV",
                    "最优缺陷浓度": "5%",
                    "电流密度提升": "2.3 倍",
                },
            }
        elif "合金" in str(exp.get("steps", [])):
            result = {
                "hypothesis_id": exp["hypothesis_id"],
                "plan_summary": f"DFT: {exp['steps'][0]}",
                "success": True,
                "final_code": "from ase.build import fcc111\n# Alloy surface...",
                "execution_output": "CuAu 合金能垒：0.38 eV\n稳定性良好\n选择性 CO 产物 >90%",
                "error_log": "",
                "key_findings": [
                    "CuAu 合金表现出最优的协同效应",
                    "反应能垒降至 0.38 eV",
                    "合金结构在 500 小时测试中保持稳定",
                    "CO 选择性超过 90%",
                ],
                "numerical_results": {
                    "反应能垒": "0.38 eV",
                    "混合焓": "-0.15 eV/atom",
                    "CO 选择性": ">90%",
                    "稳定性": "500 h",
                },
            }
        else:
            result = {
                "hypothesis_id": exp["hypothesis_id"],
                "plan_summary": f"ML: {exp['steps'][0]}",
                "success": True,
                "final_code": "from sklearn.ensemble import GradientBoostingRegressor\n# ML screening...",
                "execution_output": "模型 R^2=0.85\n筛选出 4 个候选材料\n最优预测活性超基准 40%",
                "error_log": "",
                "key_findings": [
                    "GBDT 模型 R^2 达到 0.85",
                    "筛选出 4 个高活性候选催化剂",
                    "关键描述符：d 带中心、配位数、电负性差",
                    "最优候选物活性预测超基准 40%",
                ],
                "numerical_results": {
                    "模型 R^2": "0.85",
                    "候选材料数": "4",
                    "预测活性提升": "40%",
                    "筛选效率": "6x",
                },
            }
        results.append(result)
    return results


def analyze_and_evolve(hypotheses: List[Dict], results: List[Dict], iteration: int) -> Dict:
    """分析实验结果并进化假设。"""
    evaluations = []
    analyses = []
    updated_ids = []

    verdicts = ["SUPPORTS", "SUPPORTS", "SUPPORTS", "NEEDS_MORE_EVIDENCE"]
    confidence_boosts = [0.07, 0.05, 0.04, -0.02]

    for i, (hyp, result) in enumerate(zip(hypotheses, results)):
        v_idx = min(i, len(verdicts) - 1)
        verdict = verdicts[v_idx]
        boost = confidence_boosts[v_idx]

        evaluations.append({
            "hypothesis_id": hyp["id"],
            "verdict": verdict,
            "confidence_before": hyp["confidence"],
            "confidence_after": min(0.95, hyp["confidence"] + boost),
            "reasoning": f"实验数据{'支持' if verdict == 'SUPPORTS' else '部分支持'}假设预测，"
                         f"关键指标{'达到' if verdict == 'SUPPORTS' else '接近'}预期",
        })

        analyses.append({
            "hypothesis_id": hyp["id"],
            "supports_hypothesis": verdict == "SUPPORTS",
            "confidence": min(0.95, hyp["confidence"] + boost),
            "key_findings": result.get("key_findings", [])[:3],
            "statistical_summary": f"数值结果与预测吻合度{'高' if verdict == 'SUPPORTS' else '中等'}",
            "explanation": f"{'实验结果强力支持' if verdict == 'SUPPORTS' else '实验结果部分支持'}该假设。"
                           f"{'主要数值指标均达到成功标准。' if verdict == 'SUPPORTS' else '需要进一步实验验证。'}",
        })

        if verdict == "SUPPORTS":
            updated_ids.append(hyp["id"])

    new_hypotheses = []
    eliminated_ids = []
    if iteration >= 2:
        new_hypotheses.append({
            "id": f"H{iteration}_new",
            "statement": "掺杂 + 缺陷 + 合金化的多策略协同可实现超高效 CO2 转化催化剂",
            "confidence": 0.65,
            "generation": iteration,
            "status": "active",
            "parent_ids": [h["id"] for h in hypotheses[:3]],
        })

    return {
        "evaluations": evaluations,
        "analyses": analyses,
        "updated_ids": updated_ids,
        "eliminated_ids": eliminated_ids,
        "new_hypotheses": new_hypotheses,
        "evolution_reasoning": f"第 {iteration} 轮迭代中，多数假设得到实验支持。"
                               f"掺杂和缺陷策略效果显著，合金化提供了稳定性优势。"
                               f"{'提出多策略协同的新假设。' if iteration >= 2 else ''}",
        "converged": iteration >= 3,
    }
