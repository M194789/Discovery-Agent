#!/usr/bin/env python3
"""
Demo version of the Experiment system - shows how it works without requiring external APIs.
"""

import _utf8_env  # noqa: F401  # 强制 UTF-8 输出（修复 Windows GBK 控制台 emoji 崩溃）

import asyncio
import json
import os
from datetime import datetime

# Mock data for demonstration
MOCK_HYPOTHESES = [
    {
        "id": "H1",
        "statement": "过渡金属掺杂可以降低 CO2 还原反应的能垒",
        "scientific_basis": "基于 d 带中心理论，过渡金属的 d 轨道电子结构可以调控吸附能",
        "predicted_outcome": "掺杂后的催化剂在较低过电位下实现高 CO2 转化效率",
        "confidence": 0.85,
        "generation": 1,
        "status": "active"
    },
    {
        "id": "H2",
        "statement": "表面缺陷工程可以增加活性位点密度",
        "scientific_basis": "氧空位等缺陷可以作为 CO2 分子的吸附和活化中心",
        "predicted_outcome": "富含缺陷的表面展现出更高的电流密度和选择性",
        "confidence": 0.78,
        "generation": 1,
        "status": "active"
    },
    {
        "id": "H3",
        "statement": "合金化可以协同优化电子结构和稳定性",
        "scientific_basis": "双金属效应可以同时调节活性和抗中毒能力",
        "predicted_outcome": "合金催化剂在长时间运行中保持高活性",
        "confidence": 0.72,
        "generation": 1,
        "status": "active"
    }
]

MOCK_EXPERIMENT_PLANS = [
    {
        "hypothesis_id": "H1",
        "method": "DFT",
        "tool_chain": ["pymatgen", "ase"],
        "steps": [
            "构建过渡金属掺杂的表面模型",
            "计算 CO2 吸附能",
            "计算反应路径和能垒",
            "分析电子结构（d 带中心）"
        ],
        "code_specification": "使用 pymatgen 构建表面 slab 模型，掺杂不同过渡金属...",
        "success_criteria": "找到吸附能在 -0.5 到 -1.5 eV 之间的催化剂",
        "estimated_complexity": "high"
    }
]

MOCK_EXPERIMENT_RESULTS = [
    {
        "hypothesis_id": "H1",
        "plan_summary": "DFT: 构建过渡金属掺杂的表面模型",
        "success": True,
        "final_code": "# DFT calculation code here...\nfrom ase import Atoms\n...",
        "execution_output": "吸附能计算结果：-1.23 eV (Cu 掺杂)\n能垒：0.45 eV\n符合预期!",
        "error_log": ""
    }
]


async def demo_discovery_loop(domain: str, problem: str, max_iterations: int = 1):
    """Simulate the discovery loop for demonstration."""
    
    print("\n" + "="*70)
    print(f"🔬 自主科学发现系统 - 演示模式")
    print("="*70)
    print(f"研究领域：{domain}")
    print(f"研究问题：{problem}")
    print(f"最大迭代次数：{max_iterations}")
    print(f"时间：{datetime.now().isoformat()}")
    print("="*70)
    
    # Phase 0: Problem Analysis
    print("\n【初始化阶段】")
    print("  ✓ 搜索记忆中的相关发现...")
    print("  ✓ 加载相关技能...")
    print(f"  ✓ 生成研究问题：{problem[:50]}...")
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*70}")
        print(f"📊 第 {iteration}/{max_iterations} 次迭代")
        print("="*70)
        
        # Phase A: Hypothesis Generation
        print(f"\n【Phase A】生成假设...")
        print(f"  🤖 3 个领域专家智能体正在生成假设...")
        print(f"     - 计算化学专家")
        print(f"     - 材料物理专家")
        print(f"     - 机器学习方法专家")
        
        print(f"\n  💡 生成的假设:")
        for hyp in MOCK_HYPOTHESES:
            print(f"     [{hyp['id']}] {hyp['statement']}")
            print(f"         置信度：{hyp['confidence']:.2f}")
        
        print(f"\n  🎯 开始辩论优化...")
        print(f"     原始假设：{len(MOCK_HYPOTHESES)} 个")
        print(f"     淘汰：0 个")
        print(f"     保留：{len(MOCK_HYPOTHESES)} 个")
        
        # Phase B: Experiment Planning
        print(f"\n【Phase B】规划实验...")
        for plan in MOCK_EXPERIMENT_PLANS:
            print(f"  📋 为假设 {plan['hypothesis_id']} 设计实验:")
            print(f"     方法：{plan['method']}")
            print(f"     工具：{', '.join(plan['tool_chain'])}")
            print(f"     复杂度：{plan['estimated_complexity']}")
            print(f"     步骤数：{len(plan['steps'])}")
        
        # Phase C: Experiment Execution
        print(f"\n【Phase C】执行实验...")
        for result in MOCK_EXPERIMENT_RESULTS:
            status = "✅ 成功" if result['success'] else "❌ 失败"
            print(f"  实验 {result['hypothesis_id']}: {status}")
            print(f"     输出：{result['execution_output'][:100]}...")
        
        # Phase D: Analysis & Evolution
        print(f"\n【Phase D】分析结果并进化假设...")
        print(f"  📈 分析结果:")
        print(f"     H1: SUPPORTS (置信度=0.92)")
        print(f"     H2: NEEDS_MORE_EVIDENCE (置信度=0.65)")
        print(f"     H3: REFUTES (置信度=0.30)")
        
        print(f"\n  🔄 假设进化:")
        print(f"     新增假设：1 个")
        print(f"     淘汰假设：1 个")
        print(f"     更新假设：1 个")
        
        # Phase E: Memory & Skill Update
        print(f"\n【Phase E】保存到记忆并固化技能...")
        print(f"  💾 保存发现到记忆...")
        print(f"  🔧 固化成功的实验为可复用技能...")
        
        print(f"\n  ✨ 新颖性评分更新：0.78 → 0.85")
    
    # Final Summary
    print(f"\n{'='*70}")
    print(f"🎉 发现完成!")
    print(f"{'='*70}")
    print(f"会话 ID: demo_session_001")
    print(f"总迭代次数：{max_iterations}")
    print(f"最终假设数量：3")
    print(f"支持的假设：1")
    print(f"新颖性评分：0.85")
    print(f"{'='*70}\n")
    
    # Save to disk
    output_data = {
        "session_id": "demo_session_001",
        "domain": domain,
        "problem": problem,
        "iterations": max_iterations,
        "hypotheses": MOCK_HYPOTHESES,
        "plans": MOCK_EXPERIMENT_PLANS,
        "results": MOCK_EXPERIMENT_RESULTS,
        "timestamp": datetime.now().isoformat()
    }
    
    os.makedirs("saved_code", exist_ok=True)
    output_file = "saved_code/demo_discovery.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 状态已保存到：{output_file}")
    print("\n✅ 演示完成!\n")
    
    return output_data


def main():
    """Demo entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous Scientific Discovery Demo")
    parser.add_argument("--domain", default="催化科学", help="研究领域")
    parser.add_argument("--problem", default="设计高效的二氧化碳转化催化剂", help="研究问题")
    parser.add_argument("--max-iterations", type=int, default=1, help="最大迭代次数")
    
    args = parser.parse_args()
    
    result = asyncio.run(demo_discovery_loop(
        args.domain,
        args.problem,
        args.max_iterations
    ))
    
    return 0


if __name__ == "__main__":
    exit(main())
