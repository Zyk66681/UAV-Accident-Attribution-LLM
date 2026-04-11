# 融合语义建模与大语言模型的无人机对地事故致因溯源方法
# Drone Ground Accident Causation Attribution Method

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

本项目为论文《融合语义建模与大语言模型的无人机对地事故致因溯源方法》的官方开源仓库。包含用于无人机对地事故多维推断的脱敏历史数据集，以及基于 SHAP 的可解释性与鲁棒性验证代码。

**📢 【开源计划更新说明】**
目前本仓库已率先开放核心数据集与可解释性验证代码。**关于核心的大语言模型（Qwen-7B）QLoRA 参数高效微调代码、多任务协同推断框架代码，目前正在进行脱敏与规范化整理，后续将陆续上传并完全开源，敬请期待！**

---

## 📖 项目简介 (Introduction)
在低空公共安全监管中，非结构化的无人机事故文本难以直接支撑精准的风险溯源。本项目提出了一种融合“人-机-环-管”四维语义建模与大语言模型（Qwen-7B）参数高效微调（QLoRA）的无人机对地事故自动归因方法。

## 📂 仓库结构 (Repository Structure)

* **`drone_accidents_data.xlsx`**：无人机事故原始非结构化数据集。
* **`merged_train_dataset_three_casualty_from_merged.xlsx`**：融合“人-机-环-管”结构化语义与事故后果标签的核心数据集。
* **`shap_robustness_stability.py`**：基于 SHAP 近似方法与文本局部扰动的可解释性与语义表述鲁棒性验证代码。
* **`README.md`**：项目说明文档。

## 📊 数据集说明 (Data Description)

本项目提供的数据来源于航空安全网（Aviation Safety Network, ASN）的历史记录，经过清洗与去重，共计 1213 条有效样本。

### 1. 原始数据 (`drone_accidents_data.xlsx`)
包含了原始的非结构化事故叙述（Narrative）及部分基础元数据，客观反映了事故的真实演化过程与复杂背景。核心字段包括：事故发生时间、地点、机型、飞行阶段以及事故经过的自然语言描述。

### 2. 结构化数据 (`merged_train_dataset_three_casualty_from_merged.xlsx`)
在原始数据基础上，通过语义建模抽取并重构的结构化数据，用于大模型的端到端训练与多任务推断。包含：
* **致因要素**：“人-机-环-管”（人员、设备、环境、管理）四维特征向量。
* **事故链演化逻辑**：标准化的时序事故链描述。
* **多维后果标签**：
    * 事故类型（5类：人为因素主导型、设备故障型、环境干扰型、管理缺失型、复合型）
    * 地面损害程度（4类：无损害、轻微损害、中度损害、严重损害）
    * 伤亡人数（3类区间化编码：0人、1-3人、4人及以上）

## 💻 代码说明 (Code Description)

**`shap_robustness_stability.py`** 实现了论文中提出的**归因特征可解释性与鲁棒性验证机制**。

为提升深度学习模型在公共安全关键任务中的决策透明度，该代码包含以下核心功能：
1. **SHAP 近似归因计算**：基于句子掩码扰动（Sentence Masking Perturbation）量化输入语义单元（句子级与词语级）对多维推断（事故类型、损害程度、伤亡人数）置信度的边际贡献，识别正负向驱动特征。
2. **语义表述鲁棒性验证**：采用“语义等效”的非核心词同义替换策略，计算文本局部扰动前后的 Pearson 相关系数与 Top-10 Jaccard 重合度（IoU），验证模型定责逻辑的强稳定性。

## ⚙️ 环境依赖与使用说明 (Environment & Usage)

### 运行环境
建议使用 Python 3.8 及以上版本。运行当前代码需要安装以下核心依赖库（由于数据格式为 Excel，需要安装 `openpyxl`）：
```bash
pip install torch transformers pandas numpy shap scikit-learn openpyxl


如有任何关于数据、代码或合作交流的疑问，欢迎提交 Issue，或通过邮件联系作者团队：

🏢 单位：中国人民公安大学 公共安全风险防控教育部工程研究中心 / 信息网络安全学院

📧 邮箱：202421250028@stu.ppsuc.edu.cn