# 无人机对地事故数据集字典 (UAV Ground Accident Data Dictionary)

本数据集核心源自航空安全网（ASN），经过数据清洗与标准化，共包含 1213 条有效样本，时间跨度为 2010年1月至2025年3月。

## 1. 数据预处理说明
- **剔除低质样本：** 对 `Fatalities`、`Location` 等直接影响后果推断的关键字段，剔除缺失率超过30%的样本。
- **文本规范化：** 统一专业术语，清洗非标准字符与乱码。
- **双重去重：** 基于 `accident_id` 的唯一性与 `Narrative` 的语义相似度进行去重，消除冗余干扰。
- **特征选择：** 已剔除冗余字段（如 Asn Home、URL、Download Report 等）。

## 2. 核心特征字段定义 (Table 2)

| 字段名 | 数据类型 | 字段描述 | 备注 |
| :--- | :--- | :--- | :--- |
| `accident_id` | String | 全局唯一标识符 | 用于数据去重与索引 |
| `Date,Time` | Datetime | 事故发生日期与时刻 | |
| `Location` | String | 事故地理坐标及行政区划 | 剔除高缺失率样本 |
| `Narrative` | Text | 事故经过、环境条件与潜在致因的自然语言描述 | 核心非结构化特征输入 |
| `Nature` | String | 事故性质简述（如“坠毁”） | |
| `Fatalities,Other Fatalities` | Integer | 分别记录机上人员与地面/第三方伤亡人数 | 剔除高缺失率样本 |
| `Aircraft Damage` | String | 飞行器损毁程度 | |
| `Category` | String | 事故类别 | |
| `Type` | String | 机型 | |
| `Phase` | String | 飞行阶段 | |