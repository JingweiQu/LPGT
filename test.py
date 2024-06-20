# ========================================================================================
# Test LPGT
# ========================================================================================


import torch
from pathlib import Path

from data.data_loader_graph import GMDataset, get_dataloader

from eval_fn import eval_model

from utils.config import cfg
from modules.LPModel import LPModel
from utils.utils import update_params_from_cmdline

## ***********************************************************************
## Testing
## **********************************************************************

if __name__ == "__main__":
    from utils.dup_stdout_manager import DupStdoutFileManager

    cfg = update_params_from_cmdline(default_params=cfg)
    import json
    import os

    model_dir = 'results/4/params/80/params.pt'

    with open(os.path.join(model_dir, "test_settings.json"), "w") as f:
        json.dump(cfg, f)

    with DupStdoutFileManager(str(Path(model_dir) / ("test_log.log"))) as _:
        torch.manual_seed(cfg.RANDOM_SEED)

        graph_dataset = GMDataset(cfg.DATASET_NAME, sets='test', length=None, img_resize=(256, 256))
        dataloader = get_dataloader(graph_dataset, batch_size=cfg.EVAL.BATCH_SIZE, fix_seed=True, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = LPModel(n_modules=4,
                        n_layers=3,
                        n_heads=12,
                        node_input_dim=4,
                        edge_input_dim=6,
                        node_dim=192,
                        edge_dim=192,
                        node_hid_dim=64,
                        edge_hid_dim=32,
                        output_dim=2,
                        train_fe=True,
                        normalization=True)
        model = model.cuda()

        eval_model(model, dataloader, model_path=model_dir)
