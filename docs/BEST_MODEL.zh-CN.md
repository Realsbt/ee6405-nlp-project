# 当前研究参照版本（2026-11-09）

已完成三种子的6层RoBERTa+SupCon；配置见 experiments/configs/best_confirmed.json，正式结果见 experiments/research_results/2026-11-09_confirmed_anchor。

测试Macro-F1均值 91.155848%，样本SD 1.042282%。与初始仓库均值仅约+0.00144个百分点，不能声称稳定提分；种子3407较高但42/2026下降。原始正式基线仍须保留作对照。联合增强未过预设validation门槛，最后回退6层参照。

Git仅提交代码、配方、哈希、报告和ID/概率预测；500MB级权重仍在持久资源目录，位置和SHA-256见final_manifest.json。所有后续push仅到yuzhe分支。
