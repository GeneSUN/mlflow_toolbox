from trainer import Trainer, TrainerConfig


if __name__ == "__main__":
    config = TrainerConfig()
    trainer = Trainer(config)
    trainer.train()
