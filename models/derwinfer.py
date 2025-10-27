# Please note that the current implementation of DER only contains the dynamic expansion process, since masking and pruning are not implemented by the source repo.
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from models.base import BaseLearner
from utils.inc_net import DERNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from utils.concept1_utils.infer import infer_gen
from utils.concept1_utils.utils import SyntheticImageFolder, init_synthetic_images
EPSILON = 1e-8

init_epoch = 150
init_lr = 0.001
init_milestones = [60, 120, 170]
init_lr_decay = 0.01
init_weight_decay = 0.0005


epochs = 150
lrate = 0.001
milestones = [80, 120, 150]
lrate_decay = 0.01
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2

dataset_name="etc_256"
distill_epochs=201
distill_lr=0.001
M=1
ipc=10

class DER(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = DERNet(args, False)

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        base_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        syn_dataset = SyntheticImageFolder(
            syn_root="./syn/combined",
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
        self.generate_synthetic_data(ipc=ipc, train_dataset=train_dataset)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network.train()
        if len(self._multiple_gpus) > 1 :
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                self._network_module_ptr.convnets[i].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(
                    self._total_classes - self._known_classes
                )
            else:
                self._network.weight_align(self._total_classes - self._known_classes)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self.train()
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
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_aux = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits, aux_logits = outputs["logits"], outputs["aux_logits"]
                loss_clf = F.cross_entropy(logits, targets)
                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes + 1 > 0,
                    aux_targets - self._known_classes + 1,
                    0,
                )
                loss_aux = F.cross_entropy(aux_logits, aux_targets)
                loss = loss_clf + loss_aux

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aux += loss_aux.item()
                losses_clf += loss_clf.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def generate_synthetic_data(self, ipc, train_dataset):      
        print(f"Generating synthetic data... (ipc={ipc}, total_classes={self._total_classes})")

        ipc_init = int(ipc / M / self._total_classes)
        ipc_end = ipc_init * (M + 1)
        
        self.model_list = []
        self.model_list.append(self._network)
        # if self._old_network is not None:
        #     self.model_list.append(self._old_network)
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
        base_inputs, noise_inputs = init_synthetic_images(
            num_class=self._total_classes,
            dataset=train_dataset,
            dataset_name=dataset_name,
            init_path='./syn',
            known_classes=self._known_classes,
            cur_task=self._cur_task
        )
        for ipc_id in range(ipc):
            syn = infer_gen(
                model_lists = self.model_list, 
                ipc_id = ipc_id, 
                num_class = self._total_classes, 
                iteration = distill_epochs, 
                lr = distill_lr,  
                init_path='./syn', 
                ipc_init=ipc_init, 
                base_inputs=base_inputs,
                noise_inputs=noise_inputs,
                store_best_images = True,
                dataset_name=dataset_name)
            
            # self.synthetic_data.extend(syn)
            # self.ufc.extend(aufc)
            
            #debug 
            syn_count = len(syn) if syn is not None else 0
            total_syn_count += syn_count
        # if self._old_network is not None:
        #     self._old_network.to('cpu')
        #     torch.cuda.empty_cache()
        #     print(f"   🔄 [DEBUG] Added {syn_count} synthetic samples and {aufc_count} activation features.")
        #     print(f"   📊 [DEBUG] Current totals → syn: {len(self.synthetic_data)}, aufc: {len(self.ufc)}")
        # print("\n✅ [DEBUG] Synthetic data generation complete.")
        print(f"   → Total synthetic samples generated this task: {total_syn_count}")
        # print(f"   → Cumulative synthetic data length: {len(self.synthetic_data)}")
        # print(f"   → Cumulative aufc length: {len(self.ufc)}\n")
