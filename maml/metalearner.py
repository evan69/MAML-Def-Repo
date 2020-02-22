import torch
import numpy as np

from torch.nn.utils.clip_grad import clip_grad_norm_
from maml.utils import accuracy

from advertorch.attacks import GradientSignAttack, LinfPGDAttack, CarliniWagnerL2Attack
from maml.datasets.metadataset import Task
import copy
from collections import OrderedDict

import random
from maml.models.task_net import TaskNet

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)

    return total_norm

class MetaLearner(object):
    def __init__(self, model, embedding_model, optimizers, fast_lr, loss_func,
                 first_order, num_updates, inner_loop_grad_clip,
                 collect_accuracies, device, alternating=False,
                 embedding_schedule=10, classifier_schedule=10,
                 embedding_grad_clip=0, attack_params=None, adv_train='none'):
        self._model = model
        self._embedding_model = embedding_model
        self._fast_lr = fast_lr
        self._optimizers = optimizers
        self._loss_func = loss_func
        self._first_order = first_order
        self._num_updates = num_updates
        self._inner_loop_grad_clip = inner_loop_grad_clip
        self._collect_accuracies = collect_accuracies
        self._device = device
        self._alternating = alternating
        self._alternating_schedules = (classifier_schedule, embedding_schedule)
        self._alternating_count = 0
        self._alternating_index = 1
        self._embedding_grad_clip = embedding_grad_clip
        self._attack_params = attack_params
        self._adv_train = adv_train
        self._grads_mean = []

        self.to(device)

        self._reset_measurements()

        self._adversary = self.get_adversary(self._model, attack_params=attack_params)
        # self._attack_model = copy.deepcopy(self._model)

        # for new method
        if self._adv_train == 'new':
            attack_params_list = [['FGSM', 0.1, 0],
                                  ['PGD', 0.05, 3],
                                  ['PGD', 0.01, 15],
                                  ['PGD', 0.03, 8],]
            self._adversary_list = []
            # generate adversaries for new method
            for att in attack_params_list:
                adversary_for_new = self.get_adversary(self._model, att)
                self._adversary_list.append(adversary_for_new)
            self._do_inner_adv_train = False

            self._task_net = TaskNet( input_size=self._embedding_model._embedding_dims[0],
                                      output_size=len(self._adversary_list),
                                      hidden_sizes=(64,32), disable_norm=True )
            self._task_net.cuda()
            self._task_net_optim = torch.optim.Adam(list(self._task_net.parameters()), lr=0.001)

    def get_adversary(self, model, attack_params):
        if attack_params == None:
            return
        method = attack_params[0]
        eps = attack_params[1]
        model = model.forward_single
        if method == 'FGSM':
            adversary = GradientSignAttack(model, eps=eps)
        elif method == 'PGD':
            nb_iter = attack_params[2]
            adversary = LinfPGDAttack(model, eps=eps, nb_iter=nb_iter)
        elif method == 'CW':
            adversary = CarliniWagnerL2Attack(model, num_classes=5, max_iterations=10)
        else:
            assert False
        return adversary
        # self._adversary = adversary

    def gen_adv_task(self, task, adversary):
        # self.set_adversary(self._model)
        # adv_task = Task()
        copy_task = copy.deepcopy(task)
        adv_x = adversary.perturb(copy_task.x, copy_task.y)
        adv_task = Task(adv_x, copy_task.y, 'adv_task')
        return adv_task

    def _reset_measurements(self):
        self._count_iters = 0.0
        self._cum_loss = 0.0
        self._cum_accuracy = 0.0
        # for adv
        self._adv_loss = 0.0
        self._adv_accuracy = 0.0

    def _update_measurements(self, task, loss, preds, adv_task=None, adv_loss=None, adv_preds=None):
        self._count_iters += 1.0
        self._cum_loss += loss.data.cpu().numpy()
        if self._collect_accuracies:
            self._cum_accuracy += accuracy(
                preds, task.y).data.cpu().numpy()

        if adv_task == None:
            return
        self._adv_loss += adv_loss.data.cpu().numpy()
        if self._collect_accuracies:
            self._adv_accuracy += accuracy(
                adv_preds, adv_task.y).data.cpu().numpy()

    def _pop_measurements(self):
        measurements = {}
        loss = self._cum_loss / self._count_iters
        measurements['loss'] = loss
        if self._collect_accuracies:
            accuracy = self._cum_accuracy / self._count_iters
            measurements['accuracy'] = accuracy

        # for adv
        adv_loss = self._adv_loss / self._count_iters
        measurements['adv_loss'] = adv_loss
        if self._collect_accuracies:
            adv_accuracy = self._adv_accuracy / self._count_iters
            measurements['adv_accuracy'] = adv_accuracy

        self._reset_measurements()
        return measurements

    def measure(self, tasks, train_tasks=None, adapted_params_list=None,
                embeddings_list=None):
        """Measures performance on tasks. Either train_tasks has to be a list
        of training task for computing embeddings, or adapted_params_list and
        embeddings_list have to contain adapted_params and embeddings"""
        if adapted_params_list is None:
            adapted_params_list = [None] * len(tasks)
        if embeddings_list is None:
            embeddings_list = [None] * len(tasks)
        for i in range(len(tasks)):
            params = adapted_params_list[i]
            if params is None:
                params = self._model.param_dict
            embeddings = embeddings_list[i]
            task = tasks[i]
            preds = self._model(task, params=params, embeddings=embeddings)
            loss = self._loss_func(preds, task.y)
            self._update_measurements(task, loss, preds)

        measurements = self._pop_measurements()
        return measurements

    def measure_each(self, tasks, train_tasks=None, adapted_params_list=None,
                     embeddings_list=None):
        """Measures performance on tasks. Either train_tasks has to be a list
        of training task for computing embeddings, or adapted_params_list and
        embeddings_list have to contain adapted_params and embeddings"""
        """Return a list of losses and accuracies"""
        if adapted_params_list is None:
            adapted_params_list = [None] * len(tasks)
        if embeddings_list is None:
            embeddings_list = [None] * len(tasks)
        accuracies = []
        for i in range(len(tasks)):
            params = adapted_params_list[i]
            if params is None:
                params = self._model.param_dict
            embeddings = embeddings_list[i]
            task = tasks[i]
            preds = self._model(task, params=params, embeddings=embeddings)
            pred_y = np.argmax(preds.data.cpu().numpy(), axis=-1)
            accuracy = np.mean(
                task.y.data.cpu().numpy() == 
                np.argmax(preds.data.cpu().numpy(), axis=-1))
            accuracies.append(accuracy)

        return accuracies

    def update_params(self, loss, params):
        """Apply one step of gradient descent on the loss function `loss`,
        with step-size `self._fast_lr`, and returns the updated parameters.
        """
        create_graph = not self._first_order
        grads = torch.autograd.grad(loss, params.values(),
                                    create_graph=create_graph, allow_unused=True)
        for (name, param), grad in zip(params.items(), grads):
            if self._inner_loop_grad_clip > 0 and grad is not None:
                grad = grad.clamp(min=-self._inner_loop_grad_clip,
                                  max=self._inner_loop_grad_clip)
            if grad is not None:
              params[name] = param - self._fast_lr * grad

        return params

    def adapt(self, train_tasks):
        adapted_params = []

        if self._adv_train == 'ADML' or self._adv_train == 'new':
            adv_adapted_params = [] # an extra space for adv params
        if self._adv_train == 'new':
            # decide whether do inner adv train
            self._do_inner_adv_train = (random.random() < 0.1)
            random_idxs_list = [] # list that stores pair of attak ids for each task

        embeddings_list = []

        for task in train_tasks:
            params = self._model.param_dict
            embeddings = None

            # prepare adv tasks for ADML and new
            if self._adv_train == 'ADML': # generate adv task for ADML train
                adv_params = copy.deepcopy(params)
                self._model.update_tmp_params(None) # attack current theta
                adv_task = self.gen_adv_task(task, self._adversary)
            elif self._adv_train == 'new' and self._do_inner_adv_train: # generate adv task list for new method
                adv_params = copy.deepcopy(params) # generate init params for attack #0
                self._model.update_tmp_params(None) # attack current theta for adv train
                adv_task_list = []
                random_idx_pair = random.sample(range(len(self._adversary_list)), 2)
                # for adversary in self._adversary_list[:2]: # extract 2 attacks
                for idx in random_idx_pair:
                    adversary = self._adversary_list[idx]
                    adv_task = self.gen_adv_task(task,adversary)
                    adv_task_list.append(adv_task)

            if self._embedding_model:
                embeddings = self._embedding_model(task)
                # print (embeddings)
                # print (embeddings[0].shape)
            for i in range(self._num_updates):
                preds = self._model(task, params=params, embeddings=embeddings)
                loss = self._loss_func(preds, task.y)
                if self._adv_train == 'new' and self._do_inner_adv_train: # TODO: bug here: should do adv train for each adv task
                    # attack #0 store in adv_loss
                    adv_task = adv_task_list[0]
                    adv_preds = self._model(adv_task, params=adv_params, embeddings=embeddings)
                    adv_loss = self._loss_func(adv_preds, adv_task.y)
                    adv_loss = loss + 0.2 * adv_loss

                    # attack #1 store in loss
                    adv_task = adv_task_list[1]
                    adv_preds = self._model(adv_task, params=params, embeddings=embeddings)
                    adv_loss = self._loss_func(adv_preds, adv_task.y)
                    loss = loss + 0.2 * adv_loss

                    # calculate params for attack #0 in adv_loss
                    adv_params = self.update_params(adv_loss, params=adv_params)
                params = self.update_params(loss, params=params)
                if i == 0:
                    self._update_measurements(task, loss, preds)

                if self._adv_train == 'ADML': # generate theta_adv for ADML
                    adv_preds = self._model(adv_task, params=adv_params, embeddings=embeddings)
                    adv_loss = self._loss_func(adv_preds, adv_task.y)
                    adv_params = self.update_params(adv_loss, params=adv_params)
            adapted_params.append(params)
            if self._adv_train == 'ADML' or (self._adv_train == 'new' and self._do_inner_adv_train):
            # for new method store attack #0 params
                adv_adapted_params.append(adv_params)
            if self._adv_train == 'new' and self._do_inner_adv_train:
            # update parameters of TaskNet
                task_emb = embeddings[0].detach()
                task_out = self._task_net.forward(task_emb)
                # print (task_out.shape)
                out = torch.cat((task_out[:, random_idx_pair[0]],task_out[:, random_idx_pair[1]]), 0).unsqueeze(0)
                label_0 = 1 if adv_loss < loss else 0
                ground_truth = torch.LongTensor([1-label_0]).cuda()
                # print (random_idx_pair, task_out, out, ground_truth)
                # ground_truth = torch.FloatTensor([0, ] * self._embedding_model._embedding_dims)
                # ground_truth[random_idx_pair[0]] = label_0
                # ground_truth[random_idx_pair[1]] = 1 - label_0
                cri = torch.nn.CrossEntropyLoss()
                task_net_loss = cri(out, ground_truth)
                # backward
                # print (self._task_net.state_dict()['layer1_linear.bias'])
                self._task_net_optim.zero_grad()
                task_net_loss.backward()
                self._task_net_optim.step()
                # print (self._task_net.state_dict()['layer1_linear.bias'])
            embeddings_list.append(embeddings)

        measurements = self._pop_measurements()
        if self._adv_train == 'ADML' or self._adv_train == 'new':
            # return measurements, [adapted_params, adv_adapted_params], embeddings_list
            self._adv_adapted_params = adv_adapted_params
        return measurements, adapted_params, embeddings_list

    def step(self, adapted_params_list, embeddings_list, val_tasks,
             is_training):
        for optimizer in self._optimizers:
            optimizer.zero_grad()
        post_update_losses = []

        for index, (adapted_params, embeddings, task) in enumerate(zip(
                adapted_params_list, embeddings_list, val_tasks)):
            if self._adv_train == 'ADML':
                adv_adapted_params = self._adv_adapted_params[index]
                # adv_adapted_params = adapted_params[1]
                # adapted_params = adapted_params[0]
            if self._attack_params != None:
                from collections import OrderedDict
                tmp = OrderedDict()
                for k in adapted_params.keys():
                    tmp[k] = adapted_params[k].detach()
                self._model.update_tmp_params(tmp)
                adv_task = self.gen_adv_task(task, self._adversary)
                self._model.update_tmp_params(None)
            else:
                adv_task = None
                adv_loss = None
                adv_preds = None

            preds = self._model(task, params=adapted_params,
                                embeddings=embeddings)
            loss = self._loss_func(preds, task.y)
            # post_update_losses.append(loss)
            # self._update_measurements(task, loss, preds)

            if adv_task != None:
                adv_preds = self._model(adv_task, params=adapted_params, embeddings=embeddings)
                adv_loss = self._loss_func(adv_preds, adv_task.y)
            self._update_measurements(task, loss, preds, adv_task, adv_loss, adv_preds)

            # implement Adv Querying here
            if self._adv_train == 'AdvQ':
                loss = adv_loss
            elif self._adv_train == 'ADML':
            # bug here - fixed
                ThetaC_DA_loss = adv_loss
                ac_preds = self._model(task, params=adv_adapted_params, embeddings=embeddings)
                ThetaA_DC_loss = self._loss_func(adv_preds, adv_task.y)
                loss = ThetaC_DA_loss + ThetaA_DC_loss
            elif self._adv_train == 'new':
                # pass # TODO
                loss = adv_loss
                # same as AdvQ

            post_update_losses.append(loss)

        mean_loss = torch.mean(torch.stack(post_update_losses))
        if is_training:
            mean_loss.backward()
            if self._alternating:
                self._optimizers[self._alternating_index].step()
                self._alternating_count += 1
                if self._alternating_count % self._alternating_schedules[self._alternating_index] == 0:
                    self._alternating_index = (1 - self._alternating_index)
                    self._alternating_count = 0
            else:
              self._optimizers[0].step()
              if len(self._optimizers) > 1:
                  if self._embedding_grad_clip > 0:
                      _grad_norm = clip_grad_norm_(self._embedding_model.parameters(), self._embedding_grad_clip)
                  else:
                      _grad_norm = get_grad_norm(self._embedding_model.parameters())
                  # grad_norm
                  self._grads_mean.append(_grad_norm)
                  self._optimizers[1].step()

        measurements = self._pop_measurements()
        return measurements

    def to(self, device, **kwargs):
        self._device = device
        self._model.to(device, **kwargs)
        if self._embedding_model:
            self._embedding_model.to(device, **kwargs)

    def state_dict(self):
        state = {
            'model_state_dict': self._model.state_dict(),
            'optimizers': [ optimizer.state_dict() for optimizer in self._optimizers ]
        }
        if self._embedding_model:
            state.update(
                {'embedding_model_state_dict':
                    self._embedding_model.state_dict()})
        return state
