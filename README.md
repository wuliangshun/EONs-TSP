本项目为\*\*弹性光网络信道排序问题（Channel Ordering in Elastic Optical Networks, EON）\*\*的求解代码。通过将信道排序问题转化为旅行商问题（TSP），可利用不同算法求得信道排序方案，以最大化系统性能（如最小化干扰、提升NSR等）。

## 项目结构

```
btsp_channel_ordering/
├── config.py               # 存储系统参数
├── gn_model.py             # GN模型与NSR计算函数
├── utils.py                # 工具函数，如构建U矩阵
├── solver_btsp.py          # Brute-force BTSP求解
├── visualize.py            # 可视化函数
├── main.py                 # 主运行脚本
```

## 主要功能

* 将信道排序问题建模为TSP（Traveling Salesman Problem）
* 支持Brute-force方法进行精确求解
* GN（Gaussian Noise）模型与NSR（Noise-to-Signal Ratio）计算
* 支持结果可视化
* 参数配置灵活

## 环境依赖

* Python 3.7+
* numpy
* matplotlib

如需安装依赖，可使用：

```bash
pip install -r requirements.txt
```

> 如未提供requirements.txt，可自行安装上述模块。

## 快速开始

1. **配置参数**

   根据需求修改 `config.py`，设置信道数、链路参数等系统设置。

2. **运行主程序**

   ```bash
   python main.py
   ```

   程序将自动构建U矩阵，调用BTSP求解器，计算GN模型下各信道NSR，并输出排序方案及相关性能指标。

3. **可视化结果**

   程序自动调用 `visualize.py` 展示信道排序及性能对比图。

## 代码说明

* `config.py`
  存储主要系统参数，如信道数、链路长度、信噪比等。

* `gn_model.py`
  实现GN（高斯噪声）模型和NSR计算函数。

* `utils.py`
  各类工具函数，如U矩阵生成、信道排列转换等。

* `solver_btsp.py`
  将信道排序问题建模为TSP，并采用Brute-force（暴力枚举）方法求解最优解。

* `visualize.py`
  数据可视化，包括信道排序、NSR对比等。

* `main.py`
  主运行入口，调度各模块，进行参数加载、问题建模、求解、可视化等操作。

## 适用场景

* 弹性光网络资源调度研究
* 信道分配/排序优化
* TSP建模与组合优化问题研究
* GN模型仿真与性能评估

## 致谢

如本代码对您的研究或项目有帮助，欢迎引用或致谢。

## 许可

本项目代码仅用于学术交流与研究用途，禁止商业用途。
