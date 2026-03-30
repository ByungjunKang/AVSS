import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, math, python_speech_features
from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.detectors import ContentDetector

from model.faceDetector.s3fd import S3FD
from ASD import ASD

warnings.filterwarnings("ignore")

# ==========================================================
# 1. 기존 함수의 코어 로직 유지하되, 모델을 인자로 받도록 수정
# ==========================================================

def scene_detect(args):
    videoManager = VideoManager([args.videoFilePath])
    sceneManager = SceneManager()
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source = videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    if sceneList == []:
        sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
    return sceneList

def inference_video(args, DET):
    # 수정: DET(S3FD 모델)를 매번 초기화하지 않고 인자로 받음
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
        dets.append([])
        for bbox in bboxes:
            dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})
    return dets

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def track_shot(args, sceneFaces):
    iouThres  = 0.5     
    tracks    = []
    while True:
        track     = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > args.minTrack:
            frameNum    = numpy.array([ f['frame'] for f in track ])
            bboxes      = numpy.array([numpy.array(f['bbox']) for f in track])
            frameI      = numpy.arange(frameNum[0],frameNum[-1]+1)
            bboxesI    = []
            for ij in range(0,4):
                interpfn  = interp1d(frameNum, bboxes[:,ij])
                bboxesI.append(interpfn(frameI))
            bboxesI  = numpy.stack(bboxesI, axis=1)
            if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > args.minFaceSize:
                tracks.append({'frame':frameI,'bbox':bboxesI})
    return tracks

def crop_video(args, track, cropFile):
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) 
    flist.sort()
    vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))
    dets = {'x':[], 'y':[], 's':[]}
    for det in track['bbox']: 
        dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
        dets['y'].append((det[1]+det[3])/2) 
        dets['x'].append((det[0]+det[2])/2) 
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        cs  = args.cropScale
        bs  = dets['s'][fidx]   
        bsi = int(bs * (1 + 2 * cs))  
        image = cv2.imread(flist[frame])
        frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my  = dets['y'][fidx] + bsi  
        mx  = dets['x'][fidx] + bsi  
        face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp    = cropFile + '.wav'
    audioStart  = (track['frame'][0]) / 25
    audioEnd    = (track['frame'][-1]+1) / 25
    vOut.release()
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
          (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp)) 
    subprocess.call(command, shell=True, stdout=None) 
    command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
              (cropFile, audioTmp, args.nDataLoaderThread, cropFile)) 
    subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + 't.avi')
    return {'track':track, 'proc_track':dets}

def evaluate_network_single(file, ASD_model):
    # 수정: 단일 파일만 처리하고, 외부에서 전달받은 ASD 모델을 사용
    fileName = os.path.splitext(file.split('/')[-1])[0] 
    _, audio = wavfile.read(file.replace('.avi', '.wav'))
    audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
    
    video = cv2.VideoCapture(file)
    videoFeature = []
    while video.isOpened():
        ret, frames = video.read()
        if ret == True:
            face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224,224))
            face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
            videoFeature.append(face)
        else:
            break
    video.release()
    videoFeature = numpy.array(videoFeature)
    
    length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
    audioFeature = audioFeature[:int(round(length * 100)),:]
    videoFeature = videoFeature[:int(round(length * 25)),:,:]
    
    durationSet = {1,1,1,2,2,2,3,3,4,5,6} 
    allScore = [] 
    
    for duration in durationSet:
        batchSize = int(math.ceil(length / duration))
        scores = []
        with torch.no_grad():
            for i in range(batchSize):
                inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
                inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
                embedA = ASD_model.model.forward_audio_frontend(inputA)
                embedV = ASD_model.model.forward_visual_frontend(inputV)    
                out = ASD_model.model.forward_audio_visual_backend(embedA, embedV)
                score = ASD_model.lossAV.forward(out, labels = None)
                scores.extend(score)
        allScore.append(scores)
        
    allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
    return allScore


# ==========================================================
# 2. 메인 파이프라인 (단일 비디오 -> .npy 변환)
# ==========================================================

class DummyArgs:
    # 파라미터 컨테이너 (argparse 대체)
    pass

def process_single_video(video_path, output_npy_path, DET, ASD_model):
    args = DummyArgs()
    args.videoPath = video_path
    args.nDataLoaderThread = 4
    args.facedetScale = 0.25
    args.minTrack = 10
    args.numFailedDet = 10
    args.minFaceSize = 1
    args.cropScale = 0.40
    
    # 임시 폴더 설정 (충돌 방지를 위해 비디오 이름을 포함)
    vid_name = os.path.basename(video_path).replace('.mp4', '')
    args.savePath = os.path.join('./temp_workspace', vid_name)
    
    args.pyaviPath = os.path.join(args.savePath, 'pyavi')
    args.pyframesPath = os.path.join(args.savePath, 'pyframes')
    args.pycropPath = os.path.join(args.savePath, 'pycrop')
    args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
    args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
    
    try:
        os.makedirs(args.pyaviPath, exist_ok = True) 
        os.makedirs(args.pyframesPath, exist_ok = True) 
        os.makedirs(args.pycropPath, exist_ok = True) 
        
        # 1. Video & Audio & Frames 추출
        cmd1 = f"ffmpeg -y -i {args.videoPath} -qscale:v 2 -threads {args.nDataLoaderThread} -async 1 -r 25 {args.videoFilePath} -loglevel panic"
        subprocess.call(cmd1, shell=True)
        
        cmd2 = f"ffmpeg -y -i {args.videoFilePath} -qscale:a 0 -ac 1 -vn -threads {args.nDataLoaderThread} -ar 16000 {args.audioFilePath} -loglevel panic"
        subprocess.call(cmd2, shell=True)
        
        cmd3 = f"ffmpeg -y -i {args.videoFilePath} -qscale:v 2 -threads {args.nDataLoaderThread} -f image2 {os.path.join(args.pyframesPath, '%06d.jpg')} -loglevel panic"
        subprocess.call(cmd3, shell=True)
        
        # 2. Scene & Face Detection & Tracking
        scene = scene_detect(args)
        faces = inference_video(args, DET)
        
        allTracks = []
        for shot in scene:
            if shot[1].frame_num - shot[0].frame_num >= args.minTrack: 
                allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num]))
                
        if len(allTracks) == 0:
            return False # 얼굴을 하나도 찾지 못함

        # ... (이전 코드: scene, faces 추출 및 allTracks 확보까지 동일) ...

        if len(allTracks) == 0:
            return False # 얼굴을 하나도 찾지 못함
            
        # 3. 모든 트랙 순회하며 개별 화자 추출 및 메타데이터 저장
        import pickle
        valid_tracks_info = {}
        
        for track_id, track in enumerate(allTracks):
            # 너무 짧은 궤적 (예: 25프레임 / 1초 미만)은 노이즈로 간주하고 무시
            if len(track['frame']) < 25:
                continue
                
            # 4. 개별 트랙 Crop
            crop_file_path = os.path.join(args.pycropPath, f'track_{track_id}')
            crop_video(args, track, crop_file_path)
            
            # 5. 추출된 ASD Score (1D numpy array)
            scores = evaluate_network_single(crop_file_path + '.avi', ASD_model)
            
            # 6. 최종 npy 저장 (track_id 명시)
            track_npy_path = output_npy_path.replace('.npy', f'_track{track_id}.npy')
            os.makedirs(os.path.dirname(track_npy_path), exist_ok=True)
            numpy.save(track_npy_path, scores)
            
            # 7. 시각화를 위한 바운딩 박스 메타데이터 수집
            valid_tracks_info[track_id] = {
                'frame': track['frame'],   # 프레임 인덱스 배열
                'bbox': track['bbox']      # 원본 영상 기준 [x1, y1, x2, y2] 좌표 배열
            }
        
        if not valid_tracks_info:
            return False # 필터링 후 남은 유효한 얼굴이 없음
            
        # 8. 메타데이터(.pkl) 저장
        meta_pkl_path = output_npy_path.replace('.npy', '_meta.pkl')
        with open(meta_pkl_path, 'wb') as f:
            pickle.dump(valid_tracks_info, f)
            
        return True
        
    finally:
        # 9. 디스크 용량 관리를 위해 임시 폴더 즉각 삭제
        rmtree(args.savePath, ignore_errors=True)

# ==========================================================
# 3. 실행부 (단일 GPU 루프 - 이후 멀티프로세싱으로 확장 가능)
# ==========================================================
if __name__ == '__main__':
    VOXCELEB_DIR = '/path/to/voxceleb2/dev/mp4' # 실제 데이터 경로로 변경
    OUTPUT_DIR = '/path/to/voxceleb2/asd_scores'
    WEIGHT_PATH = 'weight/pretrain_AVA.model' # LR-ASD 가중치 경로
    
    print("Loading Models (S3FD & LR-ASD)...")
    DET = S3FD(device='cuda')
    
    ASD_model = ASD()
    ASD_model.loadParameters(WEIGHT_PATH)
    ASD_model.eval()
    
    video_list = glob.glob(os.path.join(VOXCELEB_DIR, '**/*.mp4'), recursive=True)
    print(f"Total videos found: {len(video_list)}")
    
    for video_path in tqdm.tqdm(video_list):
        rel_path = os.path.relpath(video_path, VOXCELEB_DIR)
        output_npy_path = os.path.join(OUTPUT_DIR, rel_path.replace('.mp4', '.npy'))
        
        if os.path.exists(output_npy_path):
            continue
            
        success = process_single_video(video_path, output_npy_path, DET, ASD_model)
        
        if not success:
            with open('no_face_detected.txt', 'a') as f:
                f.write(video_path + '\n')
