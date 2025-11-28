import torch
import torch.nn as nn
import numpy as np
import time
import copy
from clients.clientBase import ClientBase


class clientFedDWA(ClientBase):

    def __init__(self,args, id, modelObj, train_set, test_set, **kwargs):
        super(clientFedDWA, self).__init__(args, id, modelObj, train_set, test_set, **kwargs)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.0)
        self.next_step_model = None
        self.next_round = args.next_round


    def train_one_step(self,):
        """
        train one step using the dataset(x,y) to obtain the new model parameter,
        but we don't replace the self.model by the new model parameter, we only want
        to calculate the new model parameter.
        """
        # save the old model parameter
        old_model = copy.deepcopy(self.model.state_dict())

        self.model.train()
        for e in range(self.E):
            for data in self.train_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # self.person_optimizer.zero_grad()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
        self.next_step_model = {key: copy.deepcopy(value) for key, value in self.model.named_parameters()}
        # restore the old model
        self.model.load_state_dict(old_model)

    def test_accuracy(self):
        """
        Rewrite the method in clientBase, since in the method, for each client,
        they have their personalized model, and we use the personalized model
        to test the data.
        """
        correct = 0
        total = 0
        old_model = copy.deepcopy(self.model.state_dict())
        if self.next_step_model is not None:
            cur_model = self.model.state_dict()
            for k, v in self.next_step_model.items():
                cur_model[k] = v
            self.model.load_state_dict(cur_model)

        self.model.eval()
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                _, predicts = torch.max(outputs, 1)
                correct += (predicts == labels).sum().item()
                total += len(labels)
        acc = correct / total
        self.model.load_state_dict(old_model)
        return acc

    def get_test_predictions(self):
        """
        Return true labels and predicted labels for confusion matrix
        """
        old_model = copy.deepcopy(self.model.state_dict())
        if self.next_step_model is not None:
            cur_model = self.model.state_dict()
            for k, v in self.next_step_model.items():
                cur_model[k] = v
            self.model.load_state_dict(cur_model)

        self.model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                _, predicts = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicts.cpu().numpy())
        
        self.model.load_state_dict(old_model)
        return np.array(all_labels), np.array(all_preds)


    def receive_models(self, new_state):
        """
        Rewrite the receive_models method, because the new_state is come from
        model.named_parameters()  instead of model.state_dict()
        """
        current_state = self.model.state_dict()
        for name,value in new_state.items():
            current_state[name] = value
        self.model.load_state_dict(current_state)

    def train(self):
        start_time = time.time()
        loss_logs = []
        
        # [FedVLS] Step 1: Identify Vacant Classes
        # Scan the training set to find present classes
        present_classes = set()
        # Assuming self.train_set is a Subset or Dataset where we can access targets
        # We use the safe method we added to data_utils, but here we might need to access it directly
        # or iterate if it's a custom dataset.
        # Let's try to get targets safely.
        try:
            if hasattr(self.train_set, 'dataset') and hasattr(self.train_set.dataset, 'targets'):
                 # Subset
                 targets = np.array(self.train_set.dataset.targets)[self.train_set.indices]
            elif hasattr(self.train_set, 'targets'):
                 targets = self.train_set.targets
            else:
                 # Fallback: iterate (slow but safe)
                 targets = []
                 for _, y in self.train_loader:
                     targets.extend(y.cpu().numpy())
            present_classes = set(np.unique(targets))
        except:
            # If all else fails, assume all classes are present (disable FedVLS effectively)
            present_classes = set(range(self.num_classes))
            
        all_classes = set(range(self.num_classes))
        vacant_classes = list(all_classes - present_classes)
        
        # [FedVLS] Step 2: Prepare Teacher Model
        teacher_model = None
        if self.use_fedvls and len(vacant_classes) > 0:
            teacher_model = copy.deepcopy(self.model)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
        
        self.model.train()
        for e in range(self.E):
            for data in self.train_loader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                
                # [FedVLS] Step 3: Calculate Distillation Loss
                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_outputs = teacher_model(inputs)
                    
                    # Calculate KL Divergence only for vacant classes
                    # T = 1.0 (temperature)
                    # We want to minimize KL(teacher || student) or KL(student || teacher)?
                    # Usually KL(teacher || student) -> minimize - sum(p_teacher * log(p_student))
                    # PyTorch KLDivLoss expects input as log-probs and target as probs
                    
                    log_probs_student = torch.nn.functional.log_softmax(outputs, dim=1)
                    probs_teacher = torch.nn.functional.softmax(teacher_outputs, dim=1)
                    
                    # Select only vacant classes columns
                    if len(vacant_classes) > 0:
                        # We need to mask or select specific indices
                        # loss_distill = sum( KL for c in vacant )
                        # Let's implement it manually for clarity and correctness on specific indices
                        
                        # Extract columns for vacant classes
                        vacant_indices = torch.tensor(vacant_classes).to(self.device)
                        
                        student_vacant_log_probs = log_probs_student[:, vacant_indices]
                        teacher_vacant_probs = probs_teacher[:, vacant_indices]
                        
                        # KLDivLoss: y_true * (log(y_true) - log(y_pred))
                        # But we can just use: - sum( p_teacher * log(p_student) )
                        # Since p_teacher is fixed, minimizing this is equivalent to minimizing KL
                        
                        loss_distill = - torch.sum(teacher_vacant_probs * student_vacant_log_probs) / inputs.size(0)
                        
                        loss += self.fedvls_alpha * loss_distill

                # [FedDecorr] Step 1: Calculate Decorrelation Loss
                if self.use_feddecorr:
                    features = None
                    # Try to extract features (targeting timm models like MobileViT)
                    if hasattr(self.model, 'forward_features') and hasattr(self.model, 'forward_head'):
                        try:
                            feat = self.model.forward_features(inputs)
                            features = self.model.forward_head(feat, pre_logits=True)
                        except:
                            features = None
                    
                    if features is not None:
                        # Calculate Correlation Matrix (following FedDecorr reference)
                        # features: [N, C] where N is batch size, C is feature dimension
                        N, C = features.shape
                        
                        if N > 1:  # Need at least 2 samples
                            eps = 1e-8
                            
                            # Center the features
                            features_centered = features - features.mean(dim=0, keepdim=True)
                            
                            # Normalize by variance (standard deviation)
                            features_normalized = features_centered / torch.sqrt(eps + features_centered.var(dim=0, keepdim=True))
                            
                            # Correlation matrix C = X^T * X where X is [N, C]
                            # Result is [C, C]
                            corr_matrix = torch.matmul(features_normalized.t(), features_normalized)
                            
                            # Extract off-diagonal elements
                            # Method from reference: flatten and extract off-diagonals
                            def off_diagonal(mat):
                                n, m = mat.shape
                                assert n == m
                                return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
                            
                            off_diag_elements = off_diagonal(corr_matrix)
                            
                            # Loss is mean of squared off-diagonal elements, divided by N
                            loss_decorr = (off_diag_elements.pow(2)).mean() / N
                            
                            loss += self.feddecorr_beta * loss_decorr

                loss.backward()
                # optimize
                self.optimizer.step()
                # todo: save loss
                loss_logs.append(loss.mean().item())
        # get the model parameters of the next round by training in advance
        for i in range(self.next_round):
            self.train_one_step()
        self.optimizer.zero_grad(set_to_none=True)
        
        # Clean up
        if teacher_model is not None:
            del teacher_model

        end_time = time.time()
        return np.array(loss_logs).mean()
