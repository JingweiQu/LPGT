# ========================================================================================
# evaluation function for LPGT
# ========================================================================================


import time
import numpy as np
import torch

from utils.utils import compute_pck


def eval_model(model, dataloader, model_path=None):
    print("Start evaluation...")
    since = time.time()

    print("Loading model parameters from {}".format(model_path))
    model.load_state_dict(torch.load(model_path))

    model.eval()

    ds = dataloader.dataset
    classes = ds.classes
    cls_cache = ds.cls

    all_pck = []
    category_pck = np.zeros(len(classes))
    test_time = []
    for i, cls in enumerate(classes):
        iter_num = 0

        ds.set_cls(cls)
        for inputs in dataloader:
            input_graphs = inputs['graphs'].to('cuda')
            images = inputs['images'].to('cuda')
            im_sizes = inputs['im_sizes'].to('cuda')
            L_pcks = inputs['L_pcks'].to('cuda')

            iter_num = iter_num + 1

            with torch.set_grad_enabled(False):
                start = time.time()
                output_graphs = model(input_graphs, images, 4)
                test_time += [time.time() - start]

            output_graphs_list = output_graphs.to_data_list()
            batch_pck = compute_pck(output_graphs_list, im_sizes, L_pcks)
            all_pck += batch_pck
            category_pck[i] += batch_pck[0]

        category_pck[i] /= len(ds)

    pck_avg = np.mean(np.array(all_pck))
    time_avg = np.mean(np.array(test_time))

    print("Accuracy")
    for i, cls in enumerate(classes):
        print('Class {} PCK = {:.4f}'.format(cls, category_pck[i]))
    print("Mean PCK = {:.4f}".format(pck_avg))
    print("Mean generating time = {:.6f}s".format(time_avg))

    time_elapsed = time.time() - since
    print("Evaluation complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print()

    ds.cls = cls_cache
