import wandb


def init_wandb(
    project_name: str,
    config: dict,
    run_name: str = None,
    entity: str = None,
    tags: list[str] = None,
    group: str = None,
) -> None:
    """
    Initializes a Weights & Biases run with optional tags.

    Args:
        project_name: Name of the W&B project.
        config: Dictionary of hyperparameters/configuration to log.
        run_name: Optional name for this run.
        entity: Optional W&B entity (team or user name).
        tags: Optional list of string tags for the run.
    """
    wandb.init(
        project=project_name,
        config=config,
        name=run_name,
        entity=entity,
        tags=tags,
        group=group,
        reinit=True
    )


def log_metrics(
    metrics: dict,
    step: int = None
) -> None:
    """
    Logs a dictionary of metrics to W&B.

    Args:
        metrics: A dict mapping metric names to scalar values.
        step: Optional step index for the metrics.
    """
    if step is not None:
        wandb.log(metrics, step=step)
    else:
        wandb.log(metrics)


def log_image(
    image: any,
    caption: str,
    step: int = None,
    key: str = "prediction"
) -> None:
    """
    Logs an image to W&B with a caption.

    Args:
        image: Image tensor or numpy array.
        caption: Caption for the image.
        step: Optional step index.
        key: Key under which to log the image.
    """
    wandb.log({key: [wandb.Image(image, caption=caption)]}, step=step)
