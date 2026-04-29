import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import MutableMapping
from speechbrain.processing.speech_augmentation import SpeedPerturb

def flatten_dict(d, parent_key="", sep="_"):
    """Flattens a dictionary into a single-level dictionary while preserving
    parent keys. Taken from
    `SO <https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys>`_

    Args:
        d (MutableMapping): Dictionary to be flattened.
        parent_key (str): String to use as a prefix to all subsequent keys.
        sep (str): String to use as a separator between two key levels.

    Returns:
        dict: Single-level dictionary, flattened.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class AudioLightningModule(pl.LightningModule):
    def __init__(
        self,
        audio_model=None,
        video_model=None,
        optimizer=None,
        loss_func=None,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config
        # Speed Aug
        self.speedperturb = SpeedPerturb(
            self.config["datamodule"]["data_config"]["sample_rate"],
            speeds=[95, 100, 105],
            perturb_prob=1.0
        )
        # Save lightning"s AttributeDict under self.hparams
        self.default_monitor = "val_loss/dataloader_idx_0"
        self.save_hyperparameters(self.config_to_hparams(self.config))
        # self.print(self.audio_model)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        

    def forward(self, wav, mouth=None, state=None):
        """Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        return self.audio_model(wav, state=state)

    def training_step(self, batch, batch_nb):
        # 🚀 [수정] 차원 변경 대응: targets 형태는 [B, num_chunks, C, T]
        mixtures_batch, targets_batch, _ = batch
        num_chunks = targets_batch.shape[1]
        
        total_loss = 0.0
        current_state = None # 첫 청크는 초기 상태(None)로 시작
        
        # 🚀 BPTT 순차 루프
        for chunk_idx in range(num_chunks):
            # 현재 스텝의 청크 추출 (형태: [B, C, T])
            targets = targets_batch[:, chunk_idx, :, :]
            
            new_targets = []
            min_len = -1
            
            # --- SpeedAug 적용 (현재 청크 기준) ---
            if self.config["training"]["SpeedAug"] == True:
                with torch.no_grad():
                    for i in range(targets.shape[1]):
                        new_target = self.speedperturb(targets[:, i, :])
                        new_targets.append(new_target)
                        if i == 0:
                            min_len = new_target.shape[-1]
                        else:
                            if new_target.shape[-1] < min_len:
                                min_len = new_target.shape[-1]

                targets = torch.zeros(
                            targets.shape[0],
                            targets.shape[1],
                            min_len,
                            device=targets.device,
                            dtype=torch.float,
                        )
                for i, new_target in enumerate(new_targets):
                    targets[:, i, :] = new_targets[i][:, 0:min_len]
                    
            # 믹스처 생성
            mixtures = targets.sum(1)
            
            # 🚀 모델 추론 및 State 업데이트
            est_sources, new_state = self(mixtures, state=current_state)
            
            # Loss 계산
            loss = self.loss_func["train"](est_sources, targets)
            total_loss += loss
            
            # 🚀 [매우 중요] State Detach
            # 그래디언트가 이전 청크로 계속 흘러가 OOM(메모리 초과)이 나는 것을 방지
            if new_state is not None:
                if isinstance(new_state, tuple):
                    current_state = tuple(s.detach() for s in new_state)
                elif isinstance(new_state, list):
                    current_state = [s.detach() for s in new_state]
                else:
                    current_state = new_state.detach()

        # 평균 Loss 산출
        avg_loss = total_loss / num_chunks

        self.log(
            "train_loss",
            avg_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

        return {"loss": avg_loss}


    def validation_step(self, batch, batch_nb, dataloader_idx):
        mixtures_batch, targets_batch, _ = batch
        num_chunks = targets_batch.shape[1]
        
        # --- Validation Loss 계산 ---
        if dataloader_idx == 0:
            total_val_loss = 0.0
            current_state = None
            
            for chunk_idx in range(num_chunks):
                targets = targets_batch[:, chunk_idx, :, :]
                mixtures = targets.sum(1)
                
                # State 전달 추론
                est_sources, new_state = self(mixtures, state=current_state)
                loss = self.loss_func["val"](est_sources, targets)
                total_val_loss += loss
                current_state = new_state # State 업데이트
                
            avg_val_loss = total_val_loss / num_chunks
            
            self.log("val_loss", avg_val_loss, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
            self.validation_step_outputs.append(avg_val_loss)
            return {"val_loss": avg_val_loss}

        # --- Test Loss 계산 ---
        if (self.trainer.current_epoch) % 10 == 0 and dataloader_idx == 1:
            total_test_loss = 0.0
            current_state = None
            
            for chunk_idx in range(num_chunks):
                targets = targets_batch[:, chunk_idx, :, :]
                mixtures = targets.sum(1)
                
                # State 전달 추론
                est_sources, new_state = self(mixtures, state=current_state)
                tloss = self.loss_func["val"](est_sources, targets)
                total_test_loss += tloss
                current_state = new_state # State 업데이트
                
            avg_test_loss = total_test_loss / num_chunks
            
            self.log("test_loss", avg_test_loss, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
            self.test_step_outputs.append(avg_test_loss)
            return {"test_loss": avg_test_loss}

    def on_validation_epoch_end(self):
        # val
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        val_loss = torch.mean(self.all_gather(avg_loss))
        self.log(
            "lr",
            self.optimizer.param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.logger.experiment.log(
            {"learning_rate": self.optimizer.param_groups[0]["lr"], "epoch": self.current_epoch}
        )
        self.logger.experiment.log(
            {"val_pit_sisnr": -val_loss, "epoch": self.current_epoch}
        )

        # test
        if (self.trainer.current_epoch) % 10 == 0:
            avg_loss = torch.stack(self.test_step_outputs).mean()
            test_loss = torch.mean(self.all_gather(avg_loss))
            self.logger.experiment.log(
                {"test_pit_sisnr": -test_loss, "epoch": self.current_epoch}
            )
        self.validation_step_outputs.clear()  # free memory
        self.test_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     if metric is None:
    #         scheduler.step()
    #     else:
    #         scheduler.step(metric)
    
    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return [self.val_loader, self.test_loader]

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic
