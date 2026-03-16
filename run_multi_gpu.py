import os
import glob
import math
import torch
import torch.multiprocessing as mp
import tqdm

# Step 1에서 만든 코어 함수와 모델 클래스 임포트
# (앞서 작성하신 파일명이 extract_core.py라고 가정합니다)
from extract_core import process_single_video
from model.faceDetector.s3fd import S3FD
from ASD import ASD

def worker_process(gpu_id, video_list, voxceleb_dir, output_dir, weight_path):
    """
    각 GPU에 독립적으로 할당되어 동작하는 워커 함수입니다.
    """
    # 1. 완벽한 GPU 격리 (Isolation)
    # 이 프로세스는 오직 gpu_id에 해당하는 그래픽카드만 인식합니다.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"[GPU {gpu_id}] 워커 부팅 완료! 할당된 비디오 수: {len(video_list)}")
    
    # 2. 모델 독립 적재 (매우 중요)
    # 각 프로세스 내부에서 모델을 선언해야 VRAM 충돌이 발생하지 않습니다.
    DET = S3FD(device='cuda')  # 환경변수가 설정되어 있어 자동으로 해당 GPU에 적재됨
    
    ASD_model = ASD()
    ASD_model.loadParameters(weight_path)
    ASD_model.eval().cuda()
    
    # 3. 할당된 비디오 청크 순회
    # position 인자를 통해 터미널에서 4개 GPU의 진행률 바가 겹치지 않게 출력
    for video_path in tqdm.tqdm(video_list, desc=f"GPU {gpu_id}", position=gpu_id):
        rel_path = os.path.relpath(video_path, voxceleb_dir)
        output_npy_path = os.path.join(output_dir, rel_path.replace('.mp4', '.npy'))
        
        # 이미 추출된 파일은 스킵 (중단 후 재개 시 유용)
        if os.path.exists(output_npy_path):
            continue
            
        try:
            success = process_single_video(video_path, output_npy_path, DET, ASD_model)
            
            if not success:
                # 얼굴을 찾지 못한 비디오 로깅
                with open(f'no_face_gpu_{gpu_id}.txt', 'a') as f:
                    f.write(video_path + '\n')
                    
        except Exception as e:
            # 예상치 못한 에러로 해당 GPU 워커 전체가 죽는 것을 방지
            with open(f'error_gpu_{gpu_id}.txt', 'a') as f:
                f.write(f"{video_path} | ERROR: {str(e)}\n")

def main():
    # --- 환경 설정 ---
    VOXCELEB_DIR = '/path/to/voxceleb2/dev/mp4'     # 원본 비디오 경로
    OUTPUT_DIR = '/path/to/voxceleb2/asd_scores'    # NPY 저장 경로
    WEIGHT_PATH = 'weight/pretrain_AVA.model'       # LR-ASD 가중치 경로
    NUM_GPUS = 4                                    # 사용할 GPU 개수
    
    # 1. 전체 비디오 리스트 스캔
    print("전체 비디오 리스트를 스캔 중입니다...")
    all_videos = glob.glob(os.path.join(VOXCELEB_DIR, '**/*.mp4'), recursive=True)
    
    # 2. 이미 처리된 파일 필터링 (빠른 이어하기 지원)
    # 디스크 I/O를 줄이기 위해 set 자료형 활용
    processed_files = set(glob.glob(os.path.join(OUTPUT_DIR, '**/*.npy'), recursive=True))
    pending_videos = []
    
    for v in all_videos:
        expected_npy = os.path.join(OUTPUT_DIR, os.path.relpath(v, VOXCELEB_DIR)).replace('.mp4', '.npy')
        if expected_npy not in processed_files:
            pending_videos.append(v)
            
    print(f"총 {len(all_videos)}개 중 처리할 비디오 수: {len(pending_videos)}개")
    if len(pending_videos) == 0:
        print("모든 비디오 처리가 완료되었습니다!")
        return

    # 3. 비디오 리스트를 GPU 개수에 맞게 균등 분할 (Chunking)
    chunk_size = math.ceil(len(pending_videos) / NUM_GPUS)
    video_chunks = [pending_videos[i : i + chunk_size] for i in range(0, len(pending_videos), chunk_size)]
    
    # 4. 멀티프로세싱 스폰 (Spawn)
    # fork 방식은 CUDA 환경에서 데드락을 유발하므로 반드시 spawn 사용
    mp.set_start_method('spawn', force=True)
    
    processes = []
    # 데이터가 적어 GPU 수보다 청크가 적게 쪼개졌을 경우를 대비해 min() 적용
    for gpu_id in range(min(NUM_GPUS, len(video_chunks))):
        p = mp.Process(
            target=worker_process, 
            args=(gpu_id, video_chunks[gpu_id], VOXCELEB_DIR, OUTPUT_DIR, WEIGHT_PATH)
        )
        p.start()
        processes.append(p)
        
    # 5. 모든 워커가 끝날 때까지 메인 프로세스 대기
    for p in processes:
        p.join()
        
    print("모든 GPU 워커 작업이 성공적으로 완료되었습니다!")

if __name__ == '__main__':
    main()
