import h5py
import torch
import os
import numpy as np
import copy
import time
import random
from torch import nn
import json
from pathlib import Path
from torchvision import transforms
from utils.data_utils import *
from utils.dataset import *
from torch.utils.data import DataLoader, TensorDataset

class ServerBase(object):
    def __init__(self, args, modelObj, run_times,logger):
        self.device = args.device
        self.dataset_name = args.dataset
        self.model_name = args.model
        self.algorithm_name = args.alg
        self.train_set = None
        self.test_set = None
        self.test_loader = None
        self.global_rounds = args.Tg
        self.current_global_round = 0
        self.local_steps = args.E
        self.batch_size = args.B
        self.learning_rate = args.lr
        self.evaluate_gap = 1 # how much round we will test the model
        self.times = args.times
        self.seed = args.seed

        self.sample_rate = args.sample_rate
        self.logger = logger
        self.feddwa_topk = args.feddwa_topk
        self.args = args

        self.num_types_noniid10 = args.num_types_noniid10
        self.ratio_noniid10 = args.ratio_noniid10

        self.num_clients = args.client_num
        self.client_join_ratio = args.client_frac
        self.join_clients_num = int(self.num_clients * self.client_join_ratio)
        self.algorithm_name = args.alg

        self.noniidtype = args.non_iidtype
        self.all_train_set = None
        self.all_test_set = None
        self.dirichlet_alpha = args.alpha_dir
        self.seed = args.seed
        self.num_classes = args.num_classes
        self.next_round = args.next_round


        self.global_model = copy.deepcopy(modelObj)

        self.clientsObj = []
        self.selected_clients_idx = []
        self.client_traindata_idx = []
        self.client_testdata_idx = []
        self.receive_client_models = []
        self.receive_client_datasize = []
        self.receive_client_weight = []

        self.client_test_acc_logs = []
        self.client_train_acc_logs = []
        self.client_test_loss_logs = [] # used for linear regression
        self.client_train_loss_logs = []
        self.client_mean_test_acc_logs = []
        self.global_acc_logs = []
        self.global_train_acc_logs = []
        self.round_duration_logs = []
        self.selected_clients_logs = []


    def dataset_division(self):
        # 根据模型类型决定输入尺寸
        # CNN (CIFAR10Model) 默认设计为 32x32
        # ResNet/MobileNet 等通常使用 224x224
        # MobileViT 建议使用 256x256
        if self.model_name == 'cnn':
            resize = 32
        elif self.model_name == 'mobilevit' or self.model_name == 'mobilevit_s' or self.model_name == 'fedclip':
            resize = 224
        else:
            resize = 224
            
        self.train_set, self.test_set = load_dataset(self.dataset_name, self.sample_rate, self.args.data_dir, resize=resize)
        
        # [Added] Set class prompts for FedCLIP
        if self.model_name == 'fedclip':
            class_names = None
            # Try to get class names from dataset
            if hasattr(self.train_set, 'classes'):
                class_names = self.train_set.classes
            elif hasattr(self.train_set, 'dataset') and hasattr(self.train_set.dataset, 'classes'):
                class_names = self.train_set.dataset.classes
            
            if class_names:
                self.logger.info(f"Setting FedCLIP class prompts: {class_names}")
                self.global_model.set_class_prompts(class_names)
            else:
                # Fallback for CIFAR10
                if self.dataset_name == 'cifar10tpds':
                    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                    self.global_model.set_class_prompts(class_names)
                else:
                    self.logger.warning("Could not find class names for FedCLIP prompts. Please ensure dataset has .classes attribute.")

        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
        if self.noniidtype == 8:
            noniid_fn = noniid_type8
            self.all_train_set, assignment = noniid_fn(self.dataset_name, self.train_set, num_users=self.num_clients, logger=self.logger)
            self.all_test_set, _ = noniid_fn(self.dataset_name, self.test_set, num_users=self.num_clients,
                                             sample_assignment=assignment, test=True,logger=self.logger)
        elif self.noniidtype == 9:
            noniid_fn = noniid_type9
            self.all_train_set, self.all_test_set = noniid_fn(self.dataset_name, self.train_set, self.test_set,
                                                              num_users=self.num_clients, num_classes = self.num_classes,
                                                              dirichlet_alpha=self.dirichlet_alpha, logger=self.logger)
        elif self.noniidtype == 10:
            noniid_fn = noniid_type10
            self.all_train_set = noniid_fn(self.dataset_name, self.train_set, num_users=self.num_clients,num_types=self.num_types_noniid10, ratio=self.ratio_noniid10, logger=self.logger)
            self.all_test_set = noniid_fn(self.dataset_name, self.test_set, num_users=self.num_clients, num_types=self.num_types_noniid10, ratio=self.ratio_noniid10, logger=self.logger)
        else:
            raise NotImplementedError

    def set_clients(self, args, modelObj, clientObj):
        for idx in range(self.num_clients):
            client = clientObj(args,
                               id=idx,
                               modelObj=modelObj,
                               train_set=CustomDataset(self.all_train_set[idx]),
                               test_set=CustomDataset(self.all_test_set[idx]),
                               )
            self.clientsObj.append(client)

    def select_client(self, method=0):
        """
        Two methods to select client:
        1. random manner to select client
        2. robin manner to select client (usually used in different privacy)
        """
        if method ==0: # random select
            selected_clients_idx = list(np.random.choice(range(self.num_clients), int(self.client_join_ratio * self.num_clients), replace=False))
        else: # robin manner to select
            shard_size = self.num_clients * self.client_join_ratio
            shard_num = np.ceil(1 / self.client_join_ratio)
            shard_idx = self.current_global_round % shard_num

            start = shard_idx * shard_size
            end = min((shard_idx + 1) * shard_size, self.num_clients)
            end = max(end, start + 1)
            selected_clients_idx = range(int(start), int(end))
            self.current_global_round += 1

        return selected_clients_idx

    def test_global_data(self):
        """compute accuracy using the common one test set and global model"""
        correct = 0
        total = 0
        self.global_model.eval()
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.global_model(inputs)
                _, predicts = torch.max(outputs, 1)
                correct += (predicts == labels).sum().item()
                total += len(labels)
        acc = correct / total
        # save acc
        return acc

    def get_global_predictions(self):
        """Return true labels and predicted labels for global model"""
        self.global_model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.global_model(inputs)
                _, predicts = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicts.cpu().numpy())
        return np.array(all_labels), np.array(all_preds)

    def send_models(self):
        assert (len(self.selected_clients_idx) > 0)
        for idx in self.selected_clients_idx:
            self.clientsObj[idx].receive_models(self.global_model)

    def receive_models(self):
        assert(len(self.selected_clients_idx) > 0)
        # self.receive_client_models = [copy.deepcopy(self.clientsObj[idx].model) for idx in self.selected_clients_idx]
        self.receive_client_models = [self.clientsObj[idx].model for idx in self.selected_clients_idx]
        self.receive_client_datasize = np.array([self.clientsObj[idx].data_size for idx in self.selected_clients_idx])
        self.receive_client_weight = self.receive_client_datasize/self.receive_client_datasize.sum()

    def aggregated(self):
        "Base method is FedAvg"
        assert (len(self.selected_clients_idx) > 0)
        for param in self.global_model.parameters():
            param.data.zero_()
        for weight, client_model in zip(self.receive_client_weight, self.receive_client_models):
            for global_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
                global_param.data += client_param.data.clone() * weight

    def evaluate_acc(self, selected_all=False):
        """
        calclulate each client test acccuracy and then
        calclulate the weighted-mean accuracy
        """
        if selected_all == True:
            # test all clients
            acc_logs = []
            for idx in range(self.num_clients):
                client_test_acc = self.clientsObj[idx].test_accuracy()
                acc_logs.append(client_test_acc)

            client_mean_test_acc = 0.0
            receive_client_datasize = np.array([self.clientsObj[idx].test_datasize for idx in range(self.num_clients)])
            receive_client_weight = receive_client_datasize / receive_client_datasize.sum()
            for weight, acc in zip(receive_client_weight, acc_logs):
                client_mean_test_acc += weight * acc
        else:
            acc_logs = []
            for idx in self.selected_clients_idx:
                client_test_acc = self.clientsObj[idx].test_accuracy()
                acc_logs.append(client_test_acc)

            client_mean_test_acc = 0.0
            receive_client_datasize = np.array([self.clientsObj[idx].test_datasize for idx in self.selected_clients_idx])
            receive_client_weight = receive_client_datasize / receive_client_datasize.sum()
            for weight, acc in zip(receive_client_weight, acc_logs):
                client_mean_test_acc += weight * acc

        return acc_logs, client_mean_test_acc




    def save_results(self):
        result_path = "./results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.client_test_acc_logs) > 0):
            logs = Path('./logs_feddwa')
            filename = f'{self.dataset_name}_{self.algorithm_name}_model={self.model_name}_dwaToopK={self.feddwa_topk}_next={self.next_round}_C={self.client_join_ratio}_Tg={self.global_rounds}_N={self.num_clients}_lr={self.learning_rate}_E={self.local_steps}_noniid={self.noniidtype}_nType={self.num_types_noniid10}_ratio={self.ratio_noniid10}_alpha={self.dirichlet_alpha}_{self.seed}_{self.times}.json'
            store_data = {
                'test_acc': self.client_test_acc_logs, 
                'train_loss': self.client_train_loss_logs, 
                'test_weighted-mean_acc': self.client_mean_test_acc_logs,
                'global_acc': self.global_acc_logs
            }
            with (logs / filename).open('w', encoding='utf8') as f:
                json.dump(store_data, f)
            
            # [Modified] Save results to CSV instead of .pth as requested
            import csv
            csv_filename = f'{self.dataset_name}_{self.algorithm_name}_model={self.model_name}_results.csv'
            csv_path = logs / csv_filename
            
            # Prepare data for CSV
            # Assuming all lists have the same length (number of rounds)
            rounds = range(1, len(self.client_mean_test_acc_logs) + 1)
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                # base fields
                fieldnames = ['Round', 'Global_Acc', 'Global_Train_Acc',
                              'Weighted_Mean_Acc', 'Round_Duration', 'Learning_Rate', 'Selected_Clients']

                # Add client specific columns dynamically
                num_clients = len(self.client_test_acc_logs[0]) if self.client_test_acc_logs else 0
                for i in range(num_clients):
                    fieldnames.append(f'Client_{i}_Test_Acc')
                    fieldnames.append(f'Client_{i}_Train_Acc')
                    fieldnames.append(f'Client_{i}_Train_Loss')
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for i, r in enumerate(rounds):
                    row = {
                        'Round': r,
                        'Global_Acc': self.global_acc_logs[i] if i < len(self.global_acc_logs) else None,
                        'Global_Train_Acc': self.global_train_acc_logs[i] if i < len(self.global_train_acc_logs) else None,
                        'Weighted_Mean_Acc': self.client_mean_test_acc_logs[i],
                        'Round_Duration': self.round_duration_logs[i] if i < len(self.round_duration_logs) else None,
                        'Learning_Rate': self.learning_rate,
                        'Selected_Clients': str(self.selected_clients_logs[i]) if i < len(self.selected_clients_logs) else None
                    }
                    
                    if self.client_test_acc_logs and i < len(self.client_test_acc_logs):
                        for client_idx, acc in enumerate(self.client_test_acc_logs[i]):
                            row[f'Client_{client_idx}_Test_Acc'] = acc
                    if self.client_train_acc_logs and i < len(self.client_train_acc_logs):
                        for client_idx, train_acc in enumerate(self.client_train_acc_logs[i]):
                            row[f'Client_{client_idx}_Train_Acc'] = train_acc
                            
                    if self.client_train_loss_logs and i < len(self.client_train_loss_logs):
                        for client_idx, loss in enumerate(self.client_train_loss_logs[i]):
                            row[f'Client_{client_idx}_Train_Loss'] = loss
                            
                    writer.writerow(row)
            
            self.logger.info(f"Results saved to CSV: {csv_path}")

    def save_confusion_matrices(self):
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        result_path = "./logs_feddwa/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
            
        # 1. Global Confusion Matrix
        g_labels, g_preds = self.get_global_predictions()
        cm_global = confusion_matrix(g_labels, g_preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_global, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Global Model Confusion Matrix (Acc: {np.mean(g_labels==g_preds):.4f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{result_path}/confusion_matrix_global.png')
        plt.close()
        
        # 2. Client Confusion Matrices (Save top 5 or all if small)
        # For simplicity, we save all but maybe in a separate folder if too many?
        # Let's save all for now as requested.
        cm_clients_path = os.path.join(result_path, "client_confusion_matrices")
        if not os.path.exists(cm_clients_path):
            os.makedirs(cm_clients_path)
            
        for idx, client in enumerate(self.clientsObj):
            c_labels, c_preds = client.get_test_predictions()
            cm_client = confusion_matrix(c_labels, c_preds)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_client, annot=True, fmt='d', cmap='Greens')
            plt.title(f'Client {idx} Confusion Matrix (Acc: {np.mean(c_labels==c_preds):.4f})')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'{cm_clients_path}/confusion_matrix_client_{idx}.png')
            plt.close()
            
        self.logger.info(f"Confusion matrices saved to {result_path}")

    def print(self, test_acc, train_acc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Train Accurancy: {:.4f}".format(train_acc))
        print("Average Train Loss: {:.4f}".format(train_loss))


