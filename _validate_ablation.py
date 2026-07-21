"""ablation_formal 深度有效性校验脚本"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent / 'ablation_formal'

# 期望配置（每组对应的消融配置）
EXPECTED = {
    'GroupA': ['FULL', 'A1-No-Debate', 'A2-No-GAN', 'A3-Debate-Only',
               'A4-GAN-Only', 'A5-Vanilla-LLM'],
    'GroupB': ['FULL', 'B1-No-Risk', 'B2-Static-Risk', 'B3-No-Auto-Analysis',
               'B4-Stats-Only', 'B5-No-Causal'],
    'GroupC': ['C1-No-RL', 'C2-No-Skill', 'C3-No-Cross-Domain',
               'C4-No-Literature', 'C5-No-Memory'],
    'GroupD': None,  # 不做严格名单校验，统计为主
}
DOMAINS = ['battery', 'catalysis', 'electronics']
RUNS = [1, 2, 3, 4, 5]

# 顶层 metrics 必备字段（与 ablation_metrics.extract_metrics 实际输出一致）
REQUIRED_METRICS = [
    'converged', 'iterations_used', 'novelty_score', 'experiment_success_rate',
    'mean_confidence', 'total_hypotheses', 'plan_filter_rate',
    'total_runtime_seconds', 'code_execution_failure_rate',
]
# 0..1 范围的指标（plan_filter_rate 不在此列：它是 hyp/plan 比率，可以 >1）
RATIO_METRICS = [
    'novelty_score', 'experiment_success_rate',
    'mean_confidence', 'code_execution_failure_rate',
    'statistical_significance_rate',
]

SIZE_OK = 2048    # >=2KB 视为非空壳
SIZE_HUGE = 3 * 1024 * 1024  # >3MB 视为异常膨胀（信息提示，不算失败）


def load_json(p: Path):
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f), None
    except Exception as e:
        return None, f'{type(e).__name__}: {e}'


def check_file(p: Path) -> dict:
    """单文件深度校验"""
    res = {
        'path': str(p),
        'size_kb': round(p.stat().st_size / 1024, 1),
        'ok': True,
        'issues': [],
    }
    # 1. 大小
    if p.stat().st_size < SIZE_OK:
        res['ok'] = False
        res['issues'].append(f'SIZE<2KB({res["size_kb"]}KB)')
        return res
    if p.stat().st_size > SIZE_HUGE:
        res['issues'].append(f'HUGE({res["size_kb"]}KB)')

    # 2. JSON 解析
    data, err = load_json(p)
    if err:
        res['ok'] = False
        res['issues'].append(f'JSON_PARSE_ERR: {err}')
        return res

    # 3. metrics 字段
    metrics = data.get('metrics') or {}
    missing = [k for k in REQUIRED_METRICS if k not in metrics]
    if missing:
        res['ok'] = False
        res['issues'].append(f'MISSING_METRICS: {missing}')

    # 4. metrics 数值类型快速检验
    for k in RATIO_METRICS:
        v = metrics.get(k)
        if v is not None and isinstance(v, (int, float)):
            if not (0.0 <= float(v) <= 1.0):
                res['issues'].append(f'OUT_OF_RANGE:{k}={v}')

    iters = metrics.get('iterations_used')
    if iters is not None and not (1 <= int(iters) <= 5):
        res['issues'].append(f'WEIRD_ITERS:{iters}')

    runtime = metrics.get('total_runtime_seconds')
    if runtime is not None and float(runtime) < 60.0:
        res['issues'].append(f'TOO_FAST:{runtime}s')

    # plan_filter_rate 是 hypotheses/plans 比率，合理上限给 10
    pfr = metrics.get('plan_filter_rate')
    if pfr is not None and isinstance(pfr, (int, float)):
        if not (0.0 <= float(pfr) <= 10.0):
            res['issues'].append(f'WEIRD_PFR:{pfr}')

    # 5. 其他必备顶层字段
    for k in ['config_name', 'domain', 'run_id', 'switches', 'metrics']:
        if k not in data:
            res['issues'].append(f'MISSING_TOP:{k}')

    return res


def main():
    summary = {
        'total_files': 0,
        'valid': 0,
        'invalid': 0,
        'warning_only': 0,
        'by_group': defaultdict(lambda: {
            'total': 0, 'valid': 0, 'invalid': 0,
            'expected': 0, 'missing': [],
        }),
        'invalid_files': [],
        'warning_files': [],
        'huge_files': [],
    }

    for group_dir in sorted(ROOT.iterdir()):
        if not group_dir.is_dir() or group_dir.name not in EXPECTED:
            continue
        configs = EXPECTED[group_dir.name]
        gst = summary['by_group'][group_dir.name]

        # 期望集合（GroupD 跳过严格名单）
        if configs is not None:
            for cfg in configs:
                for d in DOMAINS:
                    for r in RUNS:
                        gst['expected'] += 1
                        fn = f'{cfg}_{d}_run{r}.json'
                        if not (group_dir / fn).exists():
                            gst['missing'].append(fn)

        # 实际文件
        for jf in sorted(group_dir.glob('*.json')):
            if jf.name == 'group_summary.json':
                continue
            summary['total_files'] += 1
            gst['total'] += 1
            r = check_file(jf)
            if r['ok']:
                summary['valid'] += 1
                gst['valid'] += 1
                if r['issues']:
                    summary['warning_only'] += 1
                    summary['warning_files'].append(r)
                if any(s.startswith('HUGE') for s in r['issues']):
                    summary['huge_files'].append(r)
            else:
                summary['invalid'] += 1
                gst['invalid'] += 1
                summary['invalid_files'].append(r)

    # 输出
    print('=' * 70)
    print('ablation_formal 深度有效性校验报告')
    print('=' * 70)
    print(f'\nTotal files: {summary["total_files"]}')
    print(f'  [OK]      VALID  : {summary["valid"]}')
    print(f'  [FAIL]    INVALID: {summary["invalid"]}')
    print(f'  [WARN]    WARN ONLY (valid but with warnings): {summary["warning_only"]}')

    print('\n--- Detail by Group ---')
    for g, s in summary['by_group'].items():
        print(f'\n[{g}]')
        print(f'  expected: {s["expected"]}  actual: {s["total"]}  '
              f'valid: {s["valid"]}  invalid: {s["invalid"]}')
        if s['missing']:
            print(f'  [MISSING] ({len(s["missing"])}):')
            for fn in s['missing']:
                print(f'      - {fn}')

    if summary['invalid_files']:
        print('\n--- [FAIL] INVALID files ---')
        for r in summary['invalid_files']:
            print(f'  {Path(r["path"]).parent.name}/{Path(r["path"]).name}'
                  f'  [{r["size_kb"]}KB]  ' + '; '.join(r['issues']))

    if summary['huge_files']:
        print('\n--- [WARN] HUGE files (>3MB, valid but oversized) ---')
        for r in summary['huge_files']:
            print(f'  {Path(r["path"]).parent.name}/{Path(r["path"]).name}'
                  f'  [{r["size_kb"]}KB]')

    if summary['warning_files']:
        # 仅打印非 HUGE 的 warning
        non_huge = [r for r in summary['warning_files']
                    if not all(s.startswith('HUGE') for s in r['issues'])]
        if non_huge:
            print(f'\n--- [WARN] files ({len(non_huge)}) ---')
            for r in non_huge[:30]:
                print(f'  {Path(r["path"]).parent.name}/{Path(r["path"]).name}'
                      f'  [{r["size_kb"]}KB]  ' + '; '.join(r['issues']))
            if len(non_huge) > 30:
                print(f'  ... and {len(non_huge) - 30} more')

    print('\n' + '=' * 70)
    all_missing = sum(len(s['missing']) for s in summary['by_group'].values())
    if summary['invalid'] == 0 and all_missing == 0:
        print('[PASS] ALL files VALID, zero missing')
    else:
        print(f'[FAIL] {summary["invalid"]} INVALID, {all_missing} MISSING')
    print('=' * 70)


if __name__ == '__main__':
    main()
