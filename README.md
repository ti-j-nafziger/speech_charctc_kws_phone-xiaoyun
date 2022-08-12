---
tasks:
- keyword-spotting
domain:
- speech
frameworks:
- kaldi
backbone:
- fsmn
metrics:
- Recall/FalseAlarm
license: Apache License 2.0
tags:
- Alibaba
- kws
- CTC
- Mind Speech KWS
datasets:
  evaluation:
  - pos_testsets_phone_xiaoyun
  - neg_testsets_phone_common
widgets:
  - task: keyword-spotting
    inputs:
      - type: audio 
        name: input 
        title: 音频 
    examples:
      - name: 1
        title: 示例1 
        inputs:
          - name: input
            data: git://example/kws_xiaoyunxiaoyun.wav
    inferencespec:
      cpu: 1 #CPU数量
      memory: 1024 
---

# 语音唤醒模型介绍


## 模型描述

移动端语音唤醒模型，检测关键词为“小云小云”。模型主体为4层FSMN结构，使用CTC训练准则，参数量750K，适用于移动端设备运行。模型输入为Fbank特征，输出为基于char建模的中文全集token预测，测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。模型训练采用“basetrain + finetune”的模式，basetrain过程使用大量内部移动端数据，在此基础上，使用1万条设备端录制安静场景“小云小云”数据进行微调，得到最终面向业务的模型。后续用户可在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型，但暂时未开放模型finetune功能。


## 使用方式和范围

运行范围：
- 现阶段只能在Linux-x86_64运行，不支持Mac和Windows。

使用方式：
- 使用附带的kwsbp工具(Linux-x86_64)直接推理，分别测试正样本及负样本集合，综合选取最优工作点。

使用范围:
- 移动端设备，Android/iOS型号或版本不限，使用环境不限，采集音频为16K单通道。

目标场景:
- 移动端APP用到的关键词检测场景。


### 如何使用

- 无


#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

kwsbp_16k_pipline = pipeline(
    task=Tasks.keyword_spotting,
    model='damo/speech_charctc_kws_phone-xiaoyun')
kws_result = kwsbp_16k_pipline(audio_in='data/test/audios/kws_xiaoyunxiaoyun.wav')
```

### 模型局限性以及可能的偏差

- 考虑到正负样本测试集覆盖场景不够全面，可能有特点场合/特定人群唤醒率偏低或误唤醒偏高问题。


## 训练数据介绍

- 无


## 模型训练流程

- 无


### 预处理

- 无


## 数据评估及结果

- 模型在自建9个场景各50句的正样本集（共450条）测试，唤醒率为93.11%。在自建40小时的负样本集测试，误唤醒率为0%

