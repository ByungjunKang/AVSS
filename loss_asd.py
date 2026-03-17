import torch
import torch.nn.functional as F

def calc_si_sdr(est, ref, eps=1e-8):
    """
    단일 채널에 대한 SI-SDR 계산 (Maximize 목적이므로 음수화 전 단계)
    est, ref shape: (B, T)
    """
    # 타겟 에너지 스케일링 (Scale-Invariant)
    alpha = (ref * est).sum(dim=-1, keepdim=True) / (torch.norm(ref, dim=-1, keepdim=True)**2 + eps)
    target_scaled = alpha * ref
    
    # 노이즈(잔여물) 계산
    noise = est - target_scaled
    
    # SI-SDR (dB)
    val = (torch.norm(target_scaled, dim=-1)**2) / (torch.norm(noise, dim=-1)**2 + eps)
    si_sdr = 10 * torch.log10(val + eps)
    return si_sdr

def cal_tse_loss(est_sources, ref_sources, noise_weight=0.1):
    """
    est_sources: (B, 3, T) - [Target, Interference, Noise]
    ref_sources: (B, 3, T) - [Target, Interference, Noise]
    """
    # 1. Target Loss (채널 0): 가장 깐깐한 SI-SDR 적용
    # 음수(-)를 붙여서 Loss Minimize 문제로 변환
    loss_target = -calc_si_sdr(est_sources[:, 0, :], ref_sources[:, 0, :]).mean()
    
    # 2. Interference Loss (채널 1): 역시 깐깐한 SI-SDR 적용
    loss_interf = -calc_si_sdr(est_sources[:, 1, :], ref_sources[:, 1, :]).mean()
    
    # 3. Noise Sink Loss (채널 2): 위상에 덜 민감한 L1 Loss 적용 (앞선 실험 반영)
    loss_noise = F.l1_loss(est_sources[:, 2, :], ref_sources[:, 2, :])
    
    # 4. Total Loss (가중치 융합)
    # Target과 Interference는 동등한 비중(1.0)으로 분리력을 높이고, Noise는 0.1로 느슨하게 당김
    total_loss = loss_target + loss_interf + (noise_weight * loss_noise)
    
    # 로깅(Logging)을 위해 개별 Loss도 같이 반환하는 것이 좋습니다
    return total_loss, loss_target, loss_interf, loss_noise
