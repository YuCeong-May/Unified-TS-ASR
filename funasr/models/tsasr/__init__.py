"""Target-Speaker ASR (TS-ASR) extension for Paraformer.

本包提供 target-speaker ASR 的可插拔组件，且完全独立于原有 Paraformer 结构：
    - spk_encoder.py : 说话人 embedding 编码器（ECAPA-TDNN 等），可替换
    - fusion.py      : 说话人 embedding 与语音特征的融合方式（ADD/CAT/FiLM/CLN/CrossAttn），可替换
    - model.py       : ParaformerStreamingSpkTS，在 fbank 进入编码器前注入说话人 embedding

只要不在 config 中启用 ParaformerStreamingSpkTS / AudioDatasetTSASR，
本包不会对现有模型与训练产生任何影响。
"""
