import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import ConcatDataset,DataLoader
from torchvision import transforms
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

from utils.concept1_utils.infer import infer_gen
from utils.concept1_utils.utils import SyntheticImageFolder, init_synthetic_images

dataset_name = 'etc_256'

#distill hyperparameters
distill_lr = 0.01
first_bn_multiplier = 1.0
r_bn = 1.0
jitter = 0
ipc_start = 0
M = 2
distill_batch_size = 64
distill_epochs = 400
ipc=10

#incremental learning hyperparameters
batch_size = 128
num_workers = 4
init_epoch = 2
init_lr = 0.001
init_milestones = [60, 80]
init_lr_decay = 0.1
init_weight_decay = 0.0005
epochs = 2
lrate = 0.001
milestones = [60, 80]
lrate_decay = 0.1
weight_decay = 2e-4
T=2

class concept1(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self._old_network2 = None
        self.model_list = []

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._old_network2 = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

        self._cleanup_synthetic_folder()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        if self._old_network is not None:
            self._old_network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        base_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        syn_dataset = SyntheticImageFolder(
            syn_root="./syn",
            dataset_name=dataset_name,
            known_classes=self._known_classes,
            cur_task=self._cur_task,
            transform=transforms.Compose([*data_manager._train_trsf, *data_manager._common_trsf])
        )
        if len(syn_dataset) > 0:
            train_dataset = ConcatDataset([base_dataset, syn_dataset])
            print(f"Combined real + synthetic datasets: {len(base_dataset)} real, {len(syn_dataset)} synthetic samples.")
        else:
            train_dataset = base_dataset
            print("No synthetic data found. Using only real samples.")
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.generate_synthetic_data(self.samples_per_class, train_dataset)
        # self._construct_exemplar_random(data_manager, 10)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network2 is not None:
            self._old_network2.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=init_lr,
                # weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss_clf = F.cross_entropy(logits, targets)
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network2(inputs)["logits"],
                    T,
                )
                loss = loss_clf + loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _construct_exemplar_random(self, data_manager, m):
        logging.info("Constructing exemplars randomly for new classes...({} per classes)".format(m))
        
        for class_idx in range(self._known_classes, self._total_classes):
            # Lấy dữ liệu của class hiện tại
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            
            total_samples = len(data)
            if total_samples <= m:
                selected_indices = np.arange(total_samples)
            else:
                selected_indices = np.random.choice(total_samples, m, replace=False)
            
            selected_exemplars = data[selected_indices]
            exemplar_targets = np.full(len(selected_exemplars), class_idx)
            
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )
    
    def _cleanup_synthetic_folder(self):
        """Giữ lại đúng số ảnh mỗi class theo samples_per_class."""
        import os
        import shutil

        syn_root = "./syn"
        keep_per_class = self.samples_per_class
        if not os.path.exists(syn_root):
            return

        total_deleted = 0
        for class_dir in sorted(os.listdir(syn_root)):
            if not class_dir.startswith("new"):
                continue
            class_path = os.path.join(syn_root, class_dir)
            if not os.path.isdir(class_path):
                continue

            jpg_files = sorted(
                [f for f in os.listdir(class_path) if f.endswith(".jpg")]
            )
            if len(jpg_files) > keep_per_class:
                to_delete = jpg_files[keep_per_class:]
                for f in to_delete:
                    try:
                        os.remove(os.path.join(class_path, f))
                        total_deleted += 1
                    except Exception as e:
                        print(f"[WARN] Could not delete {f}: {e}")

        print(f"🧹 Cleaned synthetic folder: kept {keep_per_class} per class, deleted {total_deleted} extra files.")
    def generate_synthetic_data(self, ipc, train_dataset):   
        print(f"Generating synthetic data... (ipc={ipc}, total_classes={self._total_classes})")

        ipc_init = int(ipc / M / self._total_classes)
        ipc_end = ipc_init * (M + 1)
        
        self.model_list = []
        self.model_list.append(self._network)
        if self._old_network is not None:
            self.model_list.append(self._old_network)
        for model in self.model_list:
            model.eval()
            model.to("cuda")

        torch.cuda.empty_cache()
        #debug
        print(f"[DEBUG] Task {self._cur_task}: model_list contains {len(self.model_list)} model(s)")
        for idx, model in enumerate(self.model_list):
            model_name = type(model).__name__
            print(f"   - Model[{idx}]: {model_name} | device={next(model.parameters()).device}")

        total_syn_count = 0
        init_inputs = init_synthetic_images(
            num_class=self._total_classes,
            dataset=train_dataset,
            dataset_name=dataset_name,
            init_path='./syn',
            known_classes=self._known_classes
        )
        for ipc_id in range(ipc):
            syn= infer_gen(
                model_lists = self.model_list, 
                ipc_id = ipc_id, 
                num_class = self._total_classes, 
                iteration = distill_epochs, 
                lr = distill_lr,  
                init_path='./syn', 
                ipc_init=ipc_init, 
                init_inputs=init_inputs,
                store_best_images = True,
                dataset_name=dataset_name)
            
            # self.synthetic_data.extend(syn)
            # self.ufc.extend(aufc)
            
            #debug 
            syn_count = len(syn) if syn is not None else 0
            total_syn_count += syn_count
        if self._old_network is not None:
            self._old_network.to('cpu')
            torch.cuda.empty_cache()
        #     print(f"   🔄 [DEBUG] Added {syn_count} synthetic samples and {aufc_count} activation features.")
        #     print(f"   📊 [DEBUG] Current totals → syn: {len(self.synthetic_data)}, aufc: {len(self.ufc)}")
        # print("\n✅ [DEBUG] Synthetic data generation complete.")
        print(f"   → Total synthetic samples generated this task: {total_syn_count}")
        # print(f"   → Cumulative synthetic data length: {len(self.synthetic_data)}")
        # print(f"   → Cumulative aufc length: {len(self.ufc)}\n")

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]