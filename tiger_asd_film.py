import torch
import torch.nn as nn
import torch.nn.functional as F

class ASDFiLMLayer(nn.Module):
    def __init__(self, audio_channels=128):
        """
        audio_channels: TIGER Encoder를 통과한 직후의 채널 수 (보통 out_channel=128)
        """
        super().__init__()
        # 1채널(asd_score)을 받아서 gamma와 beta 2개의 파라미터를 만들기 위해 
        # 출력 채널을 audio_channels * 2 로 설정합니다. 
        # 1x1 Conv라 추가되는 파라미터와 연산량이 사실상 '0'에 가깝습니다.
        self.proj = nn.Conv1d(in_channels=1, out_channels=audio_channels * 2, kernel_size=1)
        
        # 학습 초기 안정성을 위해 가중치를 0 근처로 초기화 (선택적 최적화)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, audio_feat, asd_scores):
        """
        audio_feat: (B, C, T_audio) - TIGER Encoder 출력
        asd_scores: (B, 1, T_video) - DataLoader에서 올라온 1D Score
        """
        T_audio = audio_feat.shape[-1]
        
        # 1. Temporal Alignment (비전 25fps -> 오디오 100fps 보간)
        # linear 보간을 통해 75개의 점을 300개의 점으로 부드럽게 늘려줍니다.
        asd_aligned = F.interpolate(asd_scores, size=T_audio, mode='linear', align_corners=False)
        
        # 2. Gamma & Beta 생성
        film_params = self.proj(asd_aligned) # (B, 2*C, T_audio)
        
        # 채널 축(dim=1)을 기준으로 반으로 쪼개서 각각 gamma와 beta로 사용
        gamma, beta = torch.chunk(film_params, 2, dim=1) # 각각 (B, C, T_audio)
        
        # 3. FiLM 적용 (Modulation)
        # (1 + gamma)를 사용하는 이유: 초기화가 0으로 되어있어 학습 초반에 
        # 원본 audio_feat를 그대로 통과시키는 Identity Mapping 효과를 주기 위함입니다.
        modulated_feat = audio_feat * (1 + gamma) + beta
        
        return modulated_feat

# 기존 TIGER 모델 파일 내부

class TIGER(nn.Module):
    def __init__(self, in_channels=512, out_channels=128, num_sources=3, ...):
        super(TIGER, self).__init__()
        
        # 기존 모듈들 (이름은 look2hear 버전에 따라 조금 다를 수 있음)
        self.encoder = Encoder(...) 
        self.separator = Separator(out_channels, ...) 
        self.decoder = Decoder(...)
        
        # [추가] FiLM 모듈 초기화 (Separator가 연산할 out_channels 차원에 맞춤)
        self.film = ASDFiLMLayer(audio_channels=out_channels)

    # [수정] 입력 인자에 asd_scores 추가
    def forward(self, mix, asd_scores):
        """
        mix: (B, 1, T_wav)
        asd_scores: (B, 1, T_video)
        """
        # 1. Audio Encoding
        # x shape: (B, C, T_audio)
        x = self.encoder(mix) 
        
        # ----------------------------------------------------
        # 2. [추가] Visual Cues 주입 (FiLM 융합)
        # TIGER가 어떤 화자에 집중해야 할지 멱살을 잡고 방향을 틀어주는 역할
        # ----------------------------------------------------
        x = self.film(x, asd_scores)
        
        # 3. Separation (FFI Blocks)
        # 이미 FiLM에 의해 타겟 화자의 Feature만 강하게 증폭된 상태로 마스크 연산 돌입
        # masks shape: (B, num_sources, C, T_audio)
        masks = self.separator(x) 
        
        # 4. Decoding (3-Output: Target, Interference, Noise Sink)
        # TIGER 코드는 보통 마스크를 x와 곱한 뒤 디코더에 통과시킵니다
        est_sources = []
        for i in range(self.num_sources): # num_sources=3
            masked_feat = x * masks[:, i, :, :]
            est_wav = self.decoder(masked_feat)
            est_sources.append(est_wav)
            
        # (B, 3, T_wav) 형태로 합쳐서 반환
        return torch.cat(est_sources, dim=1) 

