import wandb


class WandbLogger:
    
    def __init__(self, exp, project=None, entity='vbutoi', name=None):
        self.exp = exp
        
        wandb.init(
            project=project,
            entity=entity,
            config=exp.config.to_dict(),
        )
        wandb.run.name = exp.path.name if name is None else name

    def __call__(self, epoch):
        df = self.exp.metrics.df
        df = df[df.epoch == epoch]
        update = {}
        for _, row in df.iterrows():
            phase = row["phase"]
            for metric in df.columns.drop('phase'):
                if metric == "epoch":
                    update["epoch"] = row["epoch"]
                else:
                    update[f"{phase}_{metric}"] = row[metric]

        wandb.log(update)