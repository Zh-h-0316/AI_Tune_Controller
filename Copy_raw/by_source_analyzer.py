import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd


def _safe_div(a, b, eps=1e-12):
    return a / np.maximum(b, eps)


def analyze_by_source_file(by_source_csv_path, output_md_path=None, top_k=10):
    """
    读取 pretrain_diagnostics_by_source_*.csv，输出问题榜单与调参建议 Markdown。
    """
    if not os.path.exists(by_source_csv_path):
        raise FileNotFoundError(f"by_source file not found: {by_source_csv_path}")

    df = pd.read_csv(by_source_csv_path)
    required_cols = {
        'split', 'source_id', 'sample_count',
        'abs_residual_mean', 'abs_model_comp_mean', 'abs_gap_mean', 'abs_gap_p99',
        'residual_mean', 'comp_mean'
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    top_k = max(1, int(top_k))

    # 核心衍生指标
    df = df.copy()
    df['comp_ratio'] = _safe_div(df['abs_model_comp_mean'].to_numpy(dtype=float), df['abs_residual_mean'].to_numpy(dtype=float))
    df['generalization_gap_ratio'] = np.nan

    # train/val 泛化差（同一 source）
    train_df = df[df['split'] == 'train'][['source_id', 'abs_gap_mean']].rename(columns={'abs_gap_mean': 'gap_train'})
    val_df = df[df['split'] == 'val'][['source_id', 'abs_gap_mean']].rename(columns={'abs_gap_mean': 'gap_val'})
    gap_join = pd.merge(train_df, val_df, on='source_id', how='inner')
    if len(gap_join) > 0:
        gap_join['generalization_gap_ratio'] = _safe_div(gap_join['gap_val'].to_numpy(dtype=float) - gap_join['gap_train'].to_numpy(dtype=float), gap_join['gap_train'].to_numpy(dtype=float))
        gap_map = dict(zip(gap_join['source_id'], gap_join['generalization_gap_ratio']))
        df['generalization_gap_ratio'] = df['source_id'].map(gap_map)

    # 问题标签
    tags = []
    for _, r in df.iterrows():
        item_tags = []
        if float(r['comp_ratio']) < 0.30:
            item_tags.append('UnderComp')
        if float(r['abs_gap_p99']) > float(r['abs_gap_mean']) * 2.0:
            item_tags.append('TailRisk')
        if float(r['residual_mean']) * float(r['comp_mean']) < 0:
            item_tags.append('DirectionBias')
        g = r['generalization_gap_ratio']
        if pd.notna(g) and float(g) > 0.20:
            item_tags.append('GeneralizationRisk')
        tags.append('|'.join(item_tags) if item_tags else 'None')
    df['issue_tags'] = tags

    # 榜单
    low_ratio = df.sort_values('comp_ratio', ascending=True).head(top_k)
    high_gap = df.sort_values('abs_gap_mean', ascending=False).head(top_k)
    high_tail = df.sort_values('abs_gap_p99', ascending=False).head(top_k)

    # 全局统计
    ratio_mean = float(df['comp_ratio'].mean())
    ratio_median = float(df['comp_ratio'].median())
    ratio_p90 = float(np.percentile(df['comp_ratio'], 90))

    gap_mean = float(df['abs_gap_mean'].mean())
    gap_p90 = float(np.percentile(df['abs_gap_mean'], 90))

    recs = []
    if ratio_mean < 0.30:
        recs.append('补偿量整体偏小：优先增大 MODE_D_DELTA_SCALE / MODE_A_BETA_SCALE，并提高 COMP_LOSS_WEIGHT。')
    elif ratio_mean < 0.60:
        recs.append('补偿量部分到位：保持补偿损失，继续提升大残差样本权重 COMP_FOCUS_GAMMA。')
    else:
        recs.append('补偿幅值整体可用：重点检查方向偏置与长尾风险。')

    if float((df['abs_gap_p99'] > df['abs_gap_mean'] * 2.0).mean()) > 0.3:
        recs.append('长尾风险较高：建议提高困难样本占比或按 source 重采样。')

    if (df['issue_tags'].str.contains('DirectionBias')).any():
        recs.append('存在方向偏置：检查误差符号定义、补偿符号以及特征归一化。')

    if pd.notna(df['generalization_gap_ratio']).any() and float((df['generalization_gap_ratio'] > 0.20).mean()) > 0.2:
        recs.append('泛化退化明显：增强数据覆盖，减少对单一 source 的依赖。')

    if output_md_path is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_md_path = os.path.join(os.path.dirname(by_source_csv_path), f'by_source_analysis_{ts}.md')

    def _to_md_table(table_df, cols):
        table_df = table_df[cols].copy()

        def _fmt(v):
            if pd.isna(v):
                return ''
            if isinstance(v, (float, np.floating)):
                return f"{float(v):.6g}"
            s = str(v)
            return s.replace('|', r'\|').replace('\n', ' ')

        headers = [str(c) for c in cols]
        lines = []
        lines.append('| ' + ' | '.join(headers) + ' |')
        lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
        for _, row in table_df.iterrows():
            values = [_fmt(row[c]) for c in cols]
            lines.append('| ' + ' | '.join(values) + ' |')
        return '\n'.join(lines)

    md = []
    md.append('# By-Source 问题榜单与调参建议')
    md.append('')
    md.append(f'- 输入文件: {by_source_csv_path}')
    md.append(f'- 样本源数量: {len(df)}')
    md.append(f'- comp_ratio(mean/median/p90): {ratio_mean:.4f} / {ratio_median:.4f} / {ratio_p90:.4f}')
    md.append(f'- abs_gap_mean(mean/p90): {gap_mean:.6f} / {gap_p90:.6f}')
    md.append('')

    md.append('## 1) 低补偿比榜单 (Top-K)')
    md.append(_to_md_table(
        low_ratio,
        ['split', 'source_id', 'sample_count', 'abs_residual_mean', 'abs_model_comp_mean', 'comp_ratio', 'abs_gap_mean', 'issue_tags']
    ))
    md.append('')

    md.append('## 2) 高缺口榜单 (Top-K)')
    md.append(_to_md_table(
        high_gap,
        ['split', 'source_id', 'sample_count', 'abs_gap_mean', 'abs_gap_p99', 'comp_ratio', 'issue_tags']
    ))
    md.append('')

    md.append('## 3) 高尾部风险榜单 (Top-K)')
    md.append(_to_md_table(
        high_tail,
        ['split', 'source_id', 'sample_count', 'abs_gap_p99', 'abs_gap_mean', 'comp_ratio', 'issue_tags']
    ))
    md.append('')

    md.append('## 4) 调参建议')
    for i, rec in enumerate(recs, start=1):
        md.append(f'{i}. {rec}')
    md.append('')

    md.append('## 5) 标签说明')
    md.append('1. UnderComp: comp_ratio < 0.30，补偿量明显不足。')
    md.append('2. TailRisk: abs_gap_p99 > 2 * abs_gap_mean，尾部风险偏高。')
    md.append('3. DirectionBias: residual_mean 与 comp_mean 符号相反。')
    md.append('4. GeneralizationRisk: 同源 val 相比 train 的 abs_gap_mean 恶化超过 20%。')
    md.append('')

    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    return {
        'output_md_path': output_md_path,
        'rows': int(len(df)),
        'ratio_mean': ratio_mean,
        'ratio_median': ratio_median,
        'ratio_p90': ratio_p90,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze pretrain_diagnostics_by_source CSV and export markdown report.')
    parser.add_argument('--input', required=True, help='Path to pretrain_diagnostics_by_source_*.csv')
    parser.add_argument('--output', default=None, help='Path to output markdown report')
    parser.add_argument('--top-k', type=int, default=10, help='Top-K rows per leaderboard')
    args = parser.parse_args()

    result = analyze_by_source_file(args.input, output_md_path=args.output, top_k=args.top_k)
    print(f"[ANALYZE] Markdown report saved: {result['output_md_path']}")
    print(
        f"[ANALYZE] rows={result['rows']}, "
        f"ratio_mean={result['ratio_mean']:.4f}, ratio_median={result['ratio_median']:.4f}, ratio_p90={result['ratio_p90']:.4f}"
    )


if __name__ == '__main__':
    main()
