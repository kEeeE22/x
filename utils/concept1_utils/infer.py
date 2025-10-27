import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import collections
import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from utils.concept1_utils.utils import clip_image, denormalize_image, BNFeatureHook, lr_cosine_policy, save_images


# Distillation hyperparams
jitter = 4
first_bn_multiplier = 10.0
r_bn = 1


def infer_gen(
    model_lists,
    ipc_id,
    num_class,
    iteration,
    lr,
    init_path,
    ipc_init,
    base_inputs,
    noise_inputs,
    store_best_images,
    dataset_name="etc_256",
    current_task=1
):
    """
    Sinh synthetic images từ base + noise cũ.
    - Với class cũ: noise khởi tạo = noise đã lưu.
    - Với class mới: noise khởi tạo = random noise.
    Lưu:
      ./syn/noise/taskXX/classXXX_noise.pt
      ./syn/combined/newXXX/classXXX_ipcXXX.jpg
    """
    print(f"[INFO] Starting infer_gen for Task {current_task}")
    syn = []
    save_every = 100

    # BN hooks
    loss_packed_features = [
        [BNFeatureHook(module) for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        for model in model_lists
    ]

    # chọn model_teacher theo ipc_id
    if len(model_lists) == 1:
        model_index = 0
    elif len(model_lists) == 2:
        half = ipc_init
        model_index = 0 if ipc_id < half else 1
    else:
        model_index = min(ipc_id // ipc_init, len(model_lists) - 1)

    model_teacher = model_lists[model_index]
    loss_r_feature_layers = loss_packed_features[model_index]
    criterion = nn.CrossEntropyLoss().cuda()

    noise_dir = f"{init_path}/noise/task{current_task:02d}"
    os.makedirs(noise_dir, exist_ok=True)

    for class_id in range(num_class):
        targets = torch.LongTensor([class_id]).to("cuda")

        base_img = base_inputs[class_id].to("cuda")

        prev_noise = noise_inputs.get(class_id, None)

        # nếu có noise cũ → dùng lại, nếu không → khởi tạo random
        if prev_noise is not None and torch.any(prev_noise != 0):
            uni_perb = prev_noise.clone().detach().requires_grad_(True)
            print(f"[LOAD NOISE] Class {class_id}: loaded previous noise.")
        else:
            uni_perb = torch.zeros_like(base_img, device="cuda", dtype=torch.float, requires_grad=True)
            print(f"[NEW NOISE] Class {class_id}: initialized random noise.")

        optimizer = optim.Adam([uni_perb], lr=lr, betas=(0.5, 0.9), eps=1e-8)
        lr_scheduler = lr_cosine_policy(lr, 0, iteration)

        for it in range(iteration):
            lr_scheduler(optimizer, it, it)
            inputs = base_img + uni_perb  # base cố định

            off1, off2 = random.randint(0, jitter), random.randint(0, jitter)
            inputs_jit = torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))

            optimizer.zero_grad()
            outputs = model_teacher(inputs_jit)["logits"]
            loss_ce = criterion(outputs, targets)

            # BN regularization loss
            rescale = [first_bn_multiplier] + [1.0] * (len(loss_r_feature_layers) - 1)
            loss_r_bn = torch.stack([
                mod.r_feature.to(loss_ce.device) * rescale[idx]
                for idx, mod in enumerate(loss_r_feature_layers)
            ]).sum()

            loss = loss_ce + r_bn * loss_r_bn
            loss.backward()
            optimizer.step()
            inputs.data = clip_image(inputs.data, dataset_name)

            if it % save_every == 0:
                print(f"[Class {class_id}] Iter {it}: CE={loss_ce.item():.3f}, BN={loss_r_bn.item():.3f}")

        final_noise = uni_perb.detach().clone().cpu()
        torch.save(final_noise, f"{noise_dir}/class{class_id:03d}_noise.pt")

        final_inputs = (base_img + uni_perb).detach().clone()
        final_img_vis = denormalize_image(final_inputs, dataset_name)

        if store_best_images:
            combined_dir = f"{init_path}/combined/new{class_id:03d}"
            os.makedirs(combined_dir, exist_ok=True)
            save_path = os.path.join(combined_dir, f"class{class_id:03d}_ipc{ipc_id:03d}.jpg")
            save_image(final_img_vis, save_path)
            print(f"[SAVE] Saved synthetic image → {save_path}")

        syn.append((final_img_vis.cpu(), targets.cpu()))

        # cleanup
        optimizer.state = collections.defaultdict(dict)
        del outputs, loss_ce, loss_r_bn, loss
        torch.cuda.empty_cache()

    # cleanup BN hooks
    for hooks in loss_packed_features:
        for h in hooks:
            h.close()
            del h
    torch.cuda.empty_cache()
    del optimizer, uni_perb
    torch.cuda.empty_cache()

    print(f"[DONE] Task {current_task} synthetic generation complete.")
    return syn