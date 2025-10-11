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
    init_inputs,
    store_best_images,
    dataset_name="etc_256"
):
    print("get_images call")
    save_every = 100
    syn= []

    loss_packed_features = [
        [BNFeatureHook(module) for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        for model in model_lists
    ]

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

    for class_id in range(num_class):
        targets = torch.LongTensor([class_id]).to("cuda")
        # is_old_class = class_id < known_classes

        # if is_old_class:
        #     jpg_dir = f"{init_path}/new{class_id:03d}"
        #     if os.path.exists(jpg_dir) and len(os.listdir(jpg_dir)) > 0:
        #         jpg_file = sorted(os.listdir(jpg_dir))[0]
        #         img_path = os.path.join(jpg_dir, jpg_file)
        #         img = Image.open(img_path).convert("L" if dataset_name == "etc_256" else "RGB")
        #         transform = transforms.ToTensor()
        #         input_original = transform(img).unsqueeze(0).to("cuda")
        #         print(f"[OLD] Loaded synthetic init for class {class_id} from {img_path}")
        #     else:
        #         print(f"[WARN] Missing jpg for class {class_id}, fallback random init")
        #         rand_idx = random.randint(0, len(dataset) - 1)
        #         _, img, _ = dataset[rand_idx]
        #         input_original = img.unsqueeze(0).to("cuda").detach()
        # else:
        #     rand_idx = random.randint(0, len(dataset) - 1)
        #     _, img, _ = dataset[rand_idx]
        #     input_original = img.unsqueeze(0).to("cuda").detach()
        #     print(f"[NEW] Random init for class {class_id}")
        input_original = init_inputs[class_id].to("cuda")
        uni_perb = torch.zeros_like(input_original, requires_grad=True, device="cuda", dtype=torch.float)
        optimizer = optim.Adam([uni_perb], lr=lr, betas=(0.5, 0.9), eps=1e-8)
        lr_scheduler = lr_cosine_policy(lr, 0, iteration)
        best_inputs, best_cost = None, 1e4

        for it in range(iteration):
            lr_scheduler(optimizer, it, it)
            inputs = input_original + uni_perb

            off1, off2 = random.randint(0, jitter), random.randint(0, jitter)
            inputs_jit = torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))

            optimizer.zero_grad()

            outputs = model_teacher(inputs_jit)["logits"]
            loss_ce = criterion(outputs, targets)

            rescale = [first_bn_multiplier] + [1.0] * (len(loss_r_feature_layers) - 1)
            loss_r_bn_feature = torch.stack([
                mod.r_feature.to(loss_ce.device) * rescale[idx]
                for idx, mod in enumerate(loss_r_feature_layers)
            ]).sum()

            loss = loss_ce + r_bn * loss_r_bn_feature
            loss.backward()
            optimizer.step()
            inputs.data = clip_image(inputs.data, dataset_name)

            # if loss.item() < best_cost:
            #     best_cost = loss.item()
            #     best_inputs = inputs.data.clone()

            if it % save_every == 0:
                print(f"[Class {class_id}] Iter {it}: CE={loss_ce.item():.3f}, BN={loss_r_bn_feature.item():.3f}")

        # if best_inputs is not None:
        #     best_inputs = denormalize_image(best_inputs, dataset_name)
        #     syn.append((best_inputs.cpu(), targets.cpu()))
        #     if store_best_images:
        #         save_images(init_path, best_inputs, targets, ipc_id)

                # Lưu .jpg cho class mới
                # if not is_old_class:
                #     dir_path = f"{init_path}/new{class_id:03d}"
                #     os.makedirs(dir_path, exist_ok=True)
                #     jpg_path = f"{dir_path}/class{class_id:03d}_ipc{ipc_id:03d}.jpg"

                #     save_image(best_inputs, jpg_path)
                #     print(f"[SAVE] Saved synthetic JPG for class {class_id} → {jpg_path}")
        final_inputs = (input_original + uni_perb).detach().clone()
        init_inputs[class_id] = final_inputs
        final_img_vis = denormalize_image(final_inputs, dataset_name)
        syn.append((final_img_vis.cpu(), targets.cpu()))
        if store_best_images:
            # Đảm bảo thư mục tồn tại
            dir_path = f"{init_path}/new{class_id:03d}"
            os.makedirs(dir_path, exist_ok=True)

            # Lưu theo task/class/ipc
            save_path = os.path.join(dir_path, f"class{class_id:03d}_ipc{ipc_id:03d}.jpg")
            save_image(final_img_vis, save_path)

            print(f"[SAVE] Saved refined synthetic image → {save_path}")
        optimizer.state = collections.defaultdict(dict)
        del outputs, loss_ce, loss_r_bn_feature, loss
        torch.cuda.empty_cache()
    for hooks in loss_packed_features:
        for h in hooks:
            h.close()
            del h
    torch.cuda.empty_cache()
    del optimizer, uni_perb
    torch.cuda.empty_cache()
    return syn
