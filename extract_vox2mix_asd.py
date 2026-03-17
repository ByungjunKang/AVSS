import os
import cv2
import numpy as np
import torch
import math
import python_speech_features
from scipy.io import wavfile
from tqdm import tqdm

# LR-ASD 모델 클래스 임포트
from ASD import ASD

def evaluate_from_face_mp4(audio_path, face_mp4_path, ASD_model):
    """
    미리 트래킹/크롭된 face mp4와 wav 파일을 읽어 1D Active Score를 반환합니다.
    """
    # ---------------------------------------------------------
    # 1. 오디오 로드 및 MFCC 추출 (100 fps)
    # ---------------------------------------------------------
    try:
        # 경고: 경로에 파일이 없거나 깨진 경우 방어 로직
        _, audio = wavfile.read(audio_path)
    except Exception as e:
        print(f"Audio Load Error: {audio_path}")
        return None
        
    audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
    
    # ---------------------------------------------------------
    # 2. Face 비디오 로드 및 전처리 (25 fps)
    # ---------------------------------------------------------
    video = cv2.VideoCapture(face_mp4_path)
    if not video.isOpened():
        print(f"Video Load Error: {face_mp4_path}")
        return None

    videoFeature = []
    while video.isOpened():
        ret, frames = video.read()
        if ret:
            # 원본 Columbia_test.py의 전처리 규격을 100% 동일하게 적용합니다.
            # 1. 흑백 변환
            face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            # 2. 224x224 리사이즈
            face = cv2.resize(face, (224, 224))
            # 3. 중앙 112x112 크롭 (입술과 코 중심)
            face = face[56:168, 56:168]
            
            videoFeature.append(face)
        else:
            break
    video.release()
    videoFeature = np.array(videoFeature)
    
    # ---------------------------------------------------------
    # 3. Audio - Video 시간축 동기화 (길이 자르기)
    # ---------------------------------------------------------
    length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
    if length <= 0:
        return None
        
    audioFeature = audioFeature[:int(round(length * 100)), :]
    videoFeature = videoFeature[:int(round(length * 25)), :, :]
    
    # ---------------------------------------------------------
    # 4. LR-ASD 추론 (원작자의 앙상블 로직 유지)
    # ---------------------------------------------------------
    durationSet = {1,1,1,2,2,2,3,3,4,5,6} 
    allScore = [] 
    
    for duration in durationSet:
        batchSize = int(math.ceil(length / duration))
        scores = []
        with torch.no_grad():
            for i in range(batchSize):
                a_clip = audioFeature[i * duration * 100 : (i+1) * duration * 100, :]
                v_clip = videoFeature[i * duration * 25 : (i+1) * duration * 25, :, :]
                
                if a_clip.shape[0] == 0 or v_clip.shape[0] == 0:
                    break
                    
                inputA = torch.FloatTensor(a_clip).unsqueeze(0).cuda()
                inputV = torch.FloatTensor(v_clip).unsqueeze(0).cuda()
                
                embedA = ASD_model.model.forward_audio_frontend(inputA)
                embedV = ASD_model.model.forward_visual_frontend(inputV)    
                out = ASD_model.model.forward_audio_visual_backend(embedA, embedV)
                score = ASD_model.lossAV.forward(out, labels = None)
                scores.extend(score)
                
        if len(scores) > 0:
            allScore.append(scores)
            
    if len(allScore) == 0:
        return None
        
    # 강건한 평균 계산을 위해 길이 맞춤
    min_len = min(len(s) for s in allScore)
    allScore = [s[:min_len] for s in allScore]
    
    final_scores = np.round((np.mean(np.array(allScore), axis = 0)), 1).astype(float)
    return final_scores

# ==========================================================
# 실행부 예시
# ==========================================================
if __name__ == '__main__':
    # 모델 초기화
    WEIGHT_PATH = 'weight/pretrain_AVA.model'
    ASD_model = ASD()
    ASD_model.loadParameters(WEIGHT_PATH)
    ASD_model.eval().cuda()
    
    # 데이터 경로 예시 (voxceleb_2mix 포맷)
    target_wav_path = '/path/to/voxceleb_2mix/audio/target_0001.wav'
    face_mp4_path = '/path/to/voxceleb_2mix/face/target_0001.mp4'
    output_npy_path = '/path/to/voxceleb_2mix/asd_scores/target_0001.npy'
    
    print("Extracting 1D Score from pre-cropped face mp4...")
    scores = evaluate_from_face_mp4(target_wav_path, face_mp4_path, ASD_model)
    
    if scores is not None:
        print(f"Extraction Success! Score Shape: {scores.shape}")
        os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
        np.save(output_npy_path, scores)
    else:
        print("Extraction Failed.")
