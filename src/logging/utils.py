import json
from types import SimpleNamespace
try:
    import wandb
except:
    print("wandb not installed.")

class BaseLogger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hparams = {}
        self.process_config(self.cfg.__dict__)

    def is_json_serializable(self, obj):
        try:
            json.dumps(obj)
            return True
        except TypeError:
            return False

    def process_config(self, cfg, parent_key=''):
        for key, value in cfg.items():
            current_key = f"{parent_key}.{key}" if parent_key else key
            
            if isinstance(value, dict):
                self.process_config(value, current_key)
            else:
                if self.is_json_serializable(value):
                    if key in self.hparams:
                        raise ValueError(f"Duplicated cfg key: {key}")
                    else:
                        self.hparams[key] = value

    def log(self, d: dict = {}, commit: bool = True):
        raise NotImplementedError("Subclasses should implement this method.")

    def finish(self):
        raise NotImplementedError("Subclasses should implement this method.")

class NoLogger(BaseLogger):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def log(self, d: dict = {}, commit: bool = True):
        return
    
    def finish(self, ):
        return

class WandbLogger(BaseLogger):
    def __init__(self, cfg):
        super().__init__(cfg)
        wandb.init(
            project= cfg.project,
            config= self.hparams,
            )

    def log(self, d: dict = {}, commit: bool = True):
        wandb.log(d, commit=commit)
        return

    def finish(self, ):
        wandb.finish()
        return

def get_logger(cfg):
    if cfg.logger == "wandb":
        logger= WandbLogger(cfg)
    else:
        logger= NoLogger(cfg)
    return logger