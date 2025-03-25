from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from nettcrstruc.dataset.dataset import MQAInferenceDataset
from nettcrstruc.utils.scoring_utils import (
    get_alphafold_rankings,
    get_plddts_for_run_dir,
    harmonic_mean,
)


def infer(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple:
    """Run inference using a model.

    Args:
        model (torch.nn.Module): Model for inference.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device to run inference on.

    Returns:
        torch.Tensor: Predictions.
        torch.Tensor: Metadata.
    """
    pred = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device, non_blocking=True)
            output = model(batch)
            pred.extend(output.detach().cpu())
    return pred


def rerank_candidates(
    run_dir: Path,
    model_list: list,
    feature_dir: Path,
    num_workers: int,
    device: torch.device,
    name: str = None,
) -> None:
    """Rerank a set of AlphaFold docking poses.

    Args:
        run_dir (Path): Path to the run directory.
        model_list (list): List of models to use for reranking.
        num_workers (int): Number of workers to use for DataLoader.
        device (torch.device): Device to use for inference.
        name: Name of the rescoring method.
    """
    scores = get_alphafold_rankings(run_dir)
    scores = scores.merge(get_plddts_for_run_dir(run_dir), on="path")

    feature_files = [
        feature_dir / f"{run_dir.stem}_ranked_{Path(p).stem.split('_')[-1]}.pt"
        for p in scores["path"]
    ]

    dataset = MQAInferenceDataset(
        feature_file_list=feature_files,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=len(scores),
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    predicted_scores = []
    for model in model_list:
        model.eval()
        y_pred = infer(model, dataloader, device)
        assert len(y_pred) == len(scores)
        predicted_scores.append(torch.stack(y_pred).cpu().numpy())


    # Rerank candidates using predicted scores
    for i in range(len(predicted_scores)):
        scores[f"pred_{i}"] = predicted_scores[i]
    scores["predicted_dockq_1"] = scores[
        [f"pred_{i}" for i in range(5)] 
    ].mean(axis=1)
    scores["predicted_dockq_2"] = scores[
        [f"pred_{i}" for i in range(5, 10)]
    ].mean(axis=1)
    scores["quality_score"] = harmonic_mean(
        [
            scores["predicted_dockq_1"],
            scores["predicted_dockq_2"],
            scores["conf"],
            scores["plddt_include_peptide_normalized"],
        ]
    )

    scores.sort_values("quality_score", ascending=False, inplace=True)
    scores.to_csv(run_dir / f"rescore_{name}.csv", index=False)


@hydra.main(version_base=None, config_path="../config", config_name="config_rescore")
def main(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_list = []
    for model_config in config.ensemble:
        model = hydra.utils.instantiate(config.ensemble[model_config].model)
        model_path = Path(__file__).resolve().parent.parent / config.ensemble[model_config].state_dict
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model_list.append(model)

    dir_list = list(Path(config.input_dir).glob("*"))
    for run_dir in tqdm(dir_list):
        if not (run_dir / f"rescore_{config.name}.csv").exists() or config.overwrite:
            rerank_candidates(
                run_dir=run_dir,
                feature_dir=Path(config.processed_dir),
                model_list=model_list,
                num_workers=config.num_workers,
                device=device,
                name=config.name,
            )
            OmegaConf.save(config, run_dir / f"rescore_config_{config.name}.yaml")


if __name__ == "__main__":
    main()
