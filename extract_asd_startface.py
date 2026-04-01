# ... (이전 코드: scene, faces 추출 후 allTracks 확보까지 동일) ...

        if len(allTracks) == 0:
            return False 
            
        # 🚀 [요구사항 1] 영상 시작 시점(Frame 0~2)부터 등장하는 화자만 필터링
        starting_tracks = []
        for t in allTracks:
            frames = t['frame'].tolist() if isinstance(t['frame'], numpy.ndarray) else t['frame']
            # 너무 짧은 노이즈 궤적 제외 & 시작 프레임이 0(또는 최대 2프레임 이내)인 궤적만 채택
            if len(frames) >= 25 and frames[0] <= 2:
                starting_tracks.append(t)
                
        if len(starting_tracks) == 0:
            print("⚠️ 시작 시점(0초)에 존재하는 얼굴 궤적이 없습니다.")
            return False 

        # 만약 시작 시점에 얼굴이 여러 개라면, 그 중에서 가장 길게 유지된 최대 2명만 선택
        starting_tracks.sort(key=lambda x: len(x['frame']), reverse=True)
        target_tracks = starting_tracks[:2] 

        # 🚀 [요구사항 2] 원본 오디오 길이(절대 시간) 기반 캔버스 생성 
        # (이 길이와 똑같이 만들어야 추론 코드에서 replicate padding이 발동하지 않음)
        from scipy.io import wavfile
        _, full_audio = wavfile.read(args.audioFilePath)
        total_audio_sec = len(full_audio) / 16000.0
        total_frames = int(total_audio_sec * 25)

        import pickle
        valid_tracks_info = {}
        
        for track_id, track in enumerate(target_tracks):
            crop_file_path = os.path.join(args.pycropPath, f'track_{track_id}')
            crop_video(args, track, crop_file_path)
            
            # 잘려진 구간에 대한 원시 점수 배열 
            scores = evaluate_network_single(crop_file_path + '.avi', ASD_model)
            
            # 🚀 [핵심] 전체 영상 길이의 캔버스를 -10.0 (확률 0%)으로 꽉 채워 초기화
            # 얼굴이 사라진 뒷부분은 이 -10.0 값이 그대로 유지되므로 '0 패딩' 효과를 냄
            global_scores = numpy.full(total_frames, -10.0, dtype=numpy.float32)
            
            frames_list = track['frame'].tolist() if isinstance(track['frame'], numpy.ndarray) else track['frame']
            
            valid_len = min(len(scores), len(frames_list))
            for i in range(valid_len):
                global_idx = frames_list[i]
                if global_idx < total_frames:
                    # 얼굴이 인식된 프레임 위치에만 실제 모델 점수를 덮어씀
                    global_scores[global_idx] = scores[i]
            
            # 완성된 타임라인 배열 저장 (길이는 항상 total_frames와 완벽 일치)
            track_npy_path = output_npy_path.replace('.npy', f'_track{track_id}.npy')
            os.makedirs(os.path.dirname(track_npy_path), exist_ok=True)
            numpy.save(track_npy_path, global_scores)
            
            valid_tracks_info[track_id] = {
                'frame': track['frame'],   
                'bbox': track['bbox']      
            }
            
        meta_pkl_path = output_npy_path.replace('.npy', '_meta.pkl')
        with open(meta_pkl_path, 'wb') as f:
            pickle.dump(valid_tracks_info, f)
            
        return True
        
    finally:
        rmtree(args.savePath, ignore_errors=True)
