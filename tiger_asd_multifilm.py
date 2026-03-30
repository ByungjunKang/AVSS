import torch
import torch.nn as nn
import torch.nn.functional as F

# 이전 대화에서 정의한 ASDFiLMLayer (Sigmoid 포함 버전)
class ASDFiLMLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # ASD Score (1D) -> gamma, beta (1D) 생성용 1x1 Conv
        self.loc = nn.Conv1d(1, channels, 1)
        self.scale = nn.Conv1d(1, channels, 1)
        
        # 가중치 초기화 (초기에는 FiLM의 영향력을 0으로 설정하여 안정적 학습 유도)
        nn.init.zeros_(self.loc.weight)
        nn.init.zeros_(self.loc.bias)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)

    def forward(self, audio_feat, asd_scores):
        """
        audio_feat: (B, C, T_audio)
        asd_scores: (B, 1, T_video) - 로짓 값 (-5~5 범위)
        """
        # [안정화] 로짓을 0~1 사이 확률로 변환
        asd_scores = torch.sigmoid(asd_scores)
        
        T_audio = audio_feat.shape[-1]
        
        # 1. Temporal Alignment (비전 25fps -> 오디오 100fps 보간)
        # asd_aligned shape: (B, 1, T_audio)
        asd_aligned = F.interpolate(asd_scores, size=T_audio, mode='linear', align_corners=False)
        
        # 2. Gamma, Beta 생성 (B, C, T_audio)
        gamma = self.scale(asd_aligned)
        beta = self.loc(asd_aligned)
        
        # 3. Feature-wise Linear Modulation
        out = audio_feat * (1 + gamma) + beta
        return out


class TIGER_ASD_MultiFiLM(nn.Module):
    def __init__(self, out_channels=128, in_channels=256, num_blocks=8, upsampling_depth=5, win=640, stride=160, num_sources=2, sample_rate=16000):
        super().__init__()
        # ... (TIGER 순정 Encoder, Bottleneck, Decoder 초기화 코드는 생략) ...
        # self.encoder = ...
        # self.bottleneck = ...
        # self.separator = ... (TCN blocks)
        # self.decoder = ...
        
        # [솔루션 1] Multi-scale FiLM 초기화
        # 1. 인코더 직후 (기존 위치)
        self.film_bottleneck = ASDFiLMLayer(out_channels) 
        
        # 2. Separator 블록들 직후, 마스크 생성기 바로 앞 (새로 추가)
        self.film_mask = ASDFiLMLayer(out_channels)
        
        # 최종 마스크 생성 레이어 (TIGER 순정)
        # self.mask_conv = nn.Conv1d(out_channels, num_sources * out_channels, 1)

    def forward(self, input_wav, asd_scores):
        """
        input_wav: (B, 1, T_audio)
        asd_scores: (B, 1, T_video)
        """
        # 1. Encoder & Bottleneck
        enc_feat = self.encoder(input_wav)
        bottleneck_feat = self.bottleneck(enc_feat) # (B, out_channels, T)
        
        # 2. 시각 힌트 1차 주입 (Bottleneck단)
        modulated_feat_1 = self.film_bottleneck(bottleneck_feat, asd_scores)
        
        # 3. Separator 블록들 (시계열 처리)
        sep_out = self.separator(modulated_feat_1) # (B, out_channels, T)
        
        # 🚀 [솔루션 1의 핵심] 시각 힌트 2차 주입 (마스크 생성 바로 앞단)
        # RNN/TCN을 거치며 흐려진 시각 조건을 다시 강력하게 부여
        modulated_feat_2 = self.film_mask(sep_out, asd_scores)
        
        # 4. 마스크 생성 및 디코딩 (TIGER 순정 로직)
        # mask = self.mask_conv(modulated_feat_2)
        # ... (이하 디코딩 및 sources cat 로직 동일) ...
        
        return est_sources
