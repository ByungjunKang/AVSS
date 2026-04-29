import os
import json
import numpy as np
from typing import Any, Tuple
import soundfile as sf
import torch
from pytorch_lightning import LightningDataModule
# from pytorch_lightning.core.mixins import HyperparametersMixin
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from typing import Dict, Iterable, List, Iterator
from rich import print
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def print_(message: str):
    print(message)


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


class MP3DDataset(Dataset):
    def __init__(
        self,
        json_dir: str = "",
        n_src: int = 2,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
    ) -> None:
        super().__init__()
        self.EPS = 1e-8
        if json_dir == None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        
        if segment is None:
            self.seg_len = None
            self.fps_len = None
        else:
            self.seg_len = int(segment * sample_rate)
            
        self.n_src = n_src
        self.test = self.seg_len is None
        mix_json = os.path.join(json_dir, "mix.json")
        sources_json = [
            os.path.join(json_dir, source + ".json") for source in ["s1", "s2"]
        ]

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))

        self.mix = []
        self.sources = []
        if self.n_src == 1:
            orig_len = len(mix_infos) * 2
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        for src_inf in sources_infos:
                            self.mix.append(mix_infos[i])
                            self.sources.append(src_inf[i])
            else:
                for i in range(len(mix_infos)):
                    for src_inf in sources_infos:
                        self.mix.append(mix_infos[i])
                        self.sources.append(src_inf[i])

            print_(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                )
            )
            self.length = len(self.mix)

        elif self.n_src == 2:
            orig_len = len(mix_infos)
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):  # Go backward
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]

            print_(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 36000, orig_len, self.seg_len
                )
            )
            self.mix = mix_infos
            self.sources = sources_infos
            self.length = len(self.mix)

    def __len__(self):
        return self.length

    def preprocess_audio_only(self, idx: int):
        # 🚀 [추가] 1. 연속된 전체 시퀀스 길이 및 청크 개수 계산
        if self.test:
            seq_len = self.mix[idx][1]
            num_chunks = 1
        else:
            # 6초 오디오에 대해 segment가 3초면 2개, 2초면 3개의 청크 계산
            num_chunks = self.mix[idx][1] // self.seg_len
            if num_chunks == 0: 
                num_chunks = 1 # 예외 처리 (seg_len보다 짧은 오디오)
            seq_len = num_chunks * self.seg_len

        # 🚀 [수정] 2. seq_len을 기준으로 랜덤 시작점 결정
        if self.mix[idx][1] == seq_len or self.test:
            rand_start = 0
        else:
            rand_start = np.random.randint(0, self.mix[idx][1] - seq_len + 1)
            
        if self.test:
            stop = None
        else:
            stop = rand_start + seq_len

        # -------------------------------------------------------------
        # Single Source (n_src == 1) 처리
        # -------------------------------------------------------------
        if self.n_src == 1:
            # Load mixture and source
            x, _ = sf.read(self.mix[idx][0], start=rand_start, stop=stop, dtype="float32")
            s, _ = sf.read(self.sources[idx][0], start=rand_start, stop=stop, dtype="float32")
            
            target = torch.from_numpy(s).unsqueeze(0) # [1, seq_len]
            mixture = torch.from_numpy(x).unsqueeze(0) # [1, seq_len]
            
            if not self.test:
                # 🚀 [핵심] [1, seq_len] -> [num_chunks, 1, seg_len] 로 Reshape
                mixture = mixture.view(1, num_chunks, self.seg_len).transpose(0, 1)
                target = target.view(1, num_chunks, self.seg_len).transpose(0, 1)
            else:
                mixture = mixture.unsqueeze(0) # [1, 1, T]
                target = target.unsqueeze(0) # [1, 1, T]
                
            if self.normalize_audio:
                # -1 차원(각 청크 내부) 기준으로 독립적인 Normalization 수행
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                target = normalize_tensor_wav(target, eps=self.EPS, std=m_std)
                
            return mixture, target, self.mix[idx][0]

        # -------------------------------------------------------------
        # Multi Source (n_src == 2) 처리
        # -------------------------------------------------------------
        if self.n_src == 2:
            # Load mixture
            x, _ = sf.read(self.mix[idx][0], start=rand_start, stop=stop, dtype="float32")
            
            # Load sources
            source_arrays = []
            for src in self.sources:
                s, _ = sf.read(src[idx][0], start=rand_start, stop=stop, dtype="float32")
                source_arrays.append(s)
                
            sources = torch.from_numpy(np.vstack(source_arrays)) # [2, seq_len]
            mixture = torch.from_numpy(x).unsqueeze(0) # [1, seq_len]

            if not self.test:
                # 🚀 [핵심] [C, seq_len] -> [num_chunks, C, seg_len] 로 Reshape
                # 1. view를 통해 시간축을 (num_chunks, seg_len)으로 쪼갬
                # 2. transpose로 num_chunks를 맨 앞(0번 차원)으로 빼냄
                mixture = mixture.view(1, num_chunks, self.seg_len).transpose(0, 1)
                sources = sources.view(self.n_src, num_chunks, self.seg_len).transpose(0, 1)
            else:
                mixture = mixture.unsqueeze(0) # [1, 1, T]
                sources = sources.unsqueeze(0) # [1, 2, T]

            if self.normalize_audio:
                # -1 차원(각 청크 내부) 기준으로 독립적인 Normalization 수행
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                sources = normalize_tensor_wav(sources, eps=self.EPS, std=m_std)

            return mixture, sources, self.mix[idx][0]

    def __getitem__(self, index: int):
        return self.preprocess_audio_only(index)


class EchoSetDataModule(object):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        n_src: int = 2,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        if train_dir == None or valid_dir == None or test_dir == None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))

        # this line allows to access init params with 'self.hparams' attribute
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.n_src = n_src
        self.sample_rate = sample_rate
        self.segment = segment
        self.normalize_audio = normalize_audio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self) -> None:
        self.data_train = MP3DDataset(
            json_dir=self.train_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=self.segment,
            normalize_audio=self.normalize_audio,
        )
        self.data_val = MP3DDataset(
            json_dir=self.valid_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=None,
            normalize_audio=self.normalize_audio,
        )
        self.data_test = MP3DDataset(
            json_dir=self.test_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=None,
            normalize_audio=self.normalize_audio,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    @property
    def make_sets(self):
        return self.data_train, self.data_val, self.data_test
