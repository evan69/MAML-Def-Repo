import torch
import numpy as np

from torch.nn.utils.clip_grad import clip_grad_norm_
from maml.utils import accuracy
from maml.utils import optimizer_to_device

from advertorch.attacks import GradientSignAttack, LinfPGDAttack, CarliniWagnerL2Attack
from advertorch.attacks import L2BasicIterativeAttack, MomentumIterativeAttack, FastFeatureAttack
from advertorch.attacks import JacobianSaliencyMapAttack, LBFGSAttack
from advertorch.attacks import SinglePixelAttack, LocalSearchAttack, LinfSPSAAttack
from maml.datasets.metadataset import Task
import copy
from collections import OrderedDict

import random
from maml.models.task_net import TaskNet
import torch.nn as nn

def random_pick(some_list, probabilities):
    x = random.uniform(0,1)
    cumulative_probability = 0.0
    for item,item_probability in zip(some_list,probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item

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
                 embedding_grad_clip=0, attack_params=None, adv_train='none', task_net=None, task_net_optim=None):
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

        if self._adv_train == 'new':
            self._task_net = task_net
            self._task_net_optim = task_net_optim

        if self._adv_train == 'Curr' or self._adv_train == 'randFT' or self._adv_train == 'advFT':
            self._smoothed_list = []
            self._adv_acc_list = []
            self._iter_cnt = 0
            self._query_strength = 0
            self._attack_params_list = [['PGD', 0.0, 1],
                                  ['PGD', 0.01, 1],
                                  ['PGD', 0.02, 3],
                                  ['PGD', 0.05, 5],
                                  ['PGD', 0.08, 8],
                                  ['PGD', 0.1,  12],
                                  ['PGD', 0.1,  15],
                                  ['PGD', 0.1,  20],
                                  ]
            self._adversary = self.get_adversary(self._model, attack_params=self._attack_params_list[self._query_strength])
            self._acc_list = []
            for item in self._attack_params_list:
                self._acc_list.append([])
            self._adversary_list = []
            # generate adversaries for Curr method
            for att in self._attack_params_list:
                adversary_for_new = self.get_adversary(self._model, att)
                self._adversary_list.append(adversary_for_new)
            if self._adv_train == 'advFT' or self._adv_train == 'randFT':
                self._adversary = self.get_adversary(self._model, attack_params=attack_params)

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
            adversary = LinfPGDAttack(model, eps=eps, nb_iter=nb_iter, rand_init=False)
        elif method == 'BIA':
            nb_iter = attack_params[2]
            adversary = L2BasicIterativeAttack(model, eps=eps, nb_iter=nb_iter)
        elif method == 'MIA':
            nb_iter = attack_params[2]
            adversary = MomentumIterativeAttack(model, eps=eps, nb_iter=nb_iter)
        elif method == 'FFA':
            nb_iter = attack_params[2]
            adversary = FastFeatureAttack(model, eps=eps, nb_iter=nb_iter)
        elif method == 'CW':
            nb_iter = attack_params[2]
            adversary = CarliniWagnerL2Attack(model, initial_const=eps, num_classes=5, max_iterations=nb_iter)
        elif method == 'JSMA':
            adversary = JacobianSaliencyMapAttack(model, num_classes=5)
            # slow, only test
        elif method == 'LBFGS':
            adversary = LBFGSAttack(model, num_classes=5)
            # slow
        elif method == 'SPA':
            adversary = SinglePixelAttack(model)
        elif method == 'LSA': # black box
            adversary = LocalSearchAttack(model)
            # slow
        elif method == 'SPSA':
            adversary = LinfSPSAAttack(model, eps=eps)
        else:
            assert False
        return adversary

    def gen_adv_task(self, task, adversary):
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

    def data_augmentation(self, task, parameter, adversary, N):
        theta_list = [parameter, ]
        task_list = [task, ]
        # print (task.x)
        for i in range(N):
            from collections import OrderedDict
            tmp = OrderedDict()
            for k in theta_list[-1].keys():
                tmp[k] = theta_list[-1][k].detach()
            self._model.update_tmp_params(tmp)
            adv_task = self.gen_adv_task(task, adversary)
            self._model.update_tmp_params(None)
            task_list.append(adv_task)
            # step 1: gen and update x

            tmp = theta_list[-1]
            preds_0 = self._model(task_list[0], params=tmp, embeddings=None)
            preds = self._model(task_list[-1], params=tmp, embeddings=None)
            loss = 0.5 * self._loss_func(preds_0, task_list[0].y) + 0.5 * self._loss_func(preds, task_list[-1].y)
            theta = self.update_params(loss, params=tmp)
            theta_list.append(theta)
            # step 2: update theta
        return task_list, theta_list

    def adapt(self, train_tasks, is_training):
        adapted_params = []

        if self._adv_train == 'ADML':
            adv_adapted_params = [] # an extra space for adv params
        if self._adv_train == 'Curr':
            self._random_strength = random.randint(0, len(self._adversary_list) - 1)
            self._random_strength = 0
        if self._adv_train == 'randFT':
            assert is_training == False
            rand_adapted_params = []

        embeddings_list = []

        for task in train_tasks:
            params = self._model.param_dict
            embeddings = None

            if self._embedding_model:
                embeddings = self._embedding_model(task)

            # prepare adv tasks for ADML and new
            if self._adv_train == 'ADML': # generate adv task for ADML train
                adv_params = copy.deepcopy(params)
                self._model.update_tmp_params(None) # attack current theta
                adv_task = self.gen_adv_task(task, self._adversary)
            elif self._adv_train == 'Curr' or self._adv_train == 'advFT':
                if not is_training:
                    self._random_strength = -1
                self._model.update_tmp_params(None) # attack current theta
                adv_task = self.gen_adv_task(task, self._adversary_list[self._random_strength])
            elif self._adv_train == 'randFT':
                assert is_training == False
                if not is_training:
                    self._random_strength = -1
                N = 6
                # DA N times
                adversary = self._adversary_list[-1]
                self._model.update_tmp_params(params)
                adv_task = self.gen_adv_task(task, adversary)
                tmp_params = copy.deepcopy(params)
                task_list, theta_list = self.data_augmentation(task=task, parameter=tmp_params, adversary=adversary, N=N)
                # generate more data via DA
                sampled_task_list = [0, 1, ] + random.sample(range(2, len(task_list)), int((N-2)*0.5))
                # sample task list for classify
                rand_params = copy.deepcopy(params)
                # for rand params

            for i in range(self._num_updates):
                preds = self._model(task, params=params, embeddings=embeddings)
                loss = self._loss_func(preds, task.y)
                if self._adv_train == 'Curr' or self._adv_train == 'advFT' or self._adv_train == 'randFT': # and self._do_inner_adv_train: # new method: reconstruct loss
                    adv_preds = self._model(adv_task, params=params, embeddings=embeddings)
                    adv_loss = self._loss_func(adv_preds, adv_task.y)
                    lam = 0.5
                    loss = (1-lam) * loss + lam * adv_loss

                params = self.update_params(loss, params=params)
                if i == 0:
                    self._update_measurements(task, loss, preds)

                if self._adv_train == 'ADML': # generate theta_adv for ADML
                    adv_preds = self._model(adv_task, params=adv_params, embeddings=embeddings)
                    adv_loss = self._loss_func(adv_preds, adv_task.y)
                    adv_params = self.update_params(adv_loss, params=adv_params)
                if self._adv_train == 'randFT':
                    rand_loss = 0.0
                    tot_coeffcient = 0.0
                    decay_u = 0.7
                    for adv_task_ids in sampled_task_list:
                        t = task_list[adv_task_ids]
                        rand_preds = self._model(t, params=rand_params, embeddings=embeddings)
                        rand_loss += (decay_u ** adv_task_ids) * self._loss_func(rand_preds, t.y)
                        tot_coeffcient += (decay_u ** adv_task_ids)
                    rand_loss /= tot_coeffcient
                    rand_params = self.update_params(rand_loss, params=rand_params)
            # params = rand_params # test

            adapted_params.append(params)
            if self._adv_train == 'ADML':# or (self._adv_train == 'new' and self._do_inner_adv_train):
                adv_adapted_params.append(adv_params)
            if self._adv_train == 'randFT':
                rand_adapted_params.append(rand_params)
            embeddings_list.append(embeddings)

        measurements = self._pop_measurements()
        if self._adv_train == 'ADML':
            self._adv_adapted_params = adv_adapted_params
        if self._adv_train == 'randFT':
            self._rand_adapted_params = rand_adapted_params
        return measurements, adapted_params, embeddings_list

    def step(self, adapted_params_list, embeddings_list, val_tasks,
             is_training):
        for optimizer in self._optimizers:
            optimizer.zero_grad()
        post_update_losses = []
        origin_losses = []

        for index, (adapted_params, embeddings, task) in enumerate(zip(
                adapted_params_list, embeddings_list, val_tasks)):
            if self._adv_train == 'ADML':
                adv_adapted_params = self._adv_adapted_params[index]
            if self._adv_train == 'randFT':
                rand_adapted_params = self._rand_adapted_params[index]

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

            if self._adv_train == 'randFT':
                adapted_params = rand_adapted_params

            preds = self._model(task, params=adapted_params,
                                embeddings=embeddings)
            loss = self._loss_func(preds, task.y)

            if adv_task != None:
                adv_preds = self._model(adv_task, params=adapted_params, embeddings=embeddings)
                adv_loss = self._loss_func(adv_preds, adv_task.y)
            self._update_measurements(task, loss, preds, adv_task, adv_loss, adv_preds)

            # implement Adv Querying here
            if self._adv_train == 'AdvQ':
                loss = adv_loss
            elif self._adv_train == 'ADML':
                ThetaC_DA_loss = adv_loss
                ac_preds = self._model(task, params=adv_adapted_params, embeddings=embeddings)
                ThetaA_DC_loss = self._loss_func(adv_preds, adv_task.y)
                loss = ThetaC_DA_loss + ThetaA_DC_loss
            elif self._adv_train == 'AdvQ-rew':
                loss = adv_loss
                # same as AdvQ
                reweight = 1.0
                gamma = 2.0
                rho = 5.0
                class_num = int(max(task.y) + 1)
                batch_size = int(task.y.shape[0])
                label = torch.LongTensor(batch_size, 1).random_() % class_num
                label = task.y.cpu().view(batch_size, 1)
                one_hot = torch.zeros(batch_size, class_num).scatter_(1, label, 1)

                softmax = nn.Softmax()

                preds_cp = softmax(preds.detach().cpu())
                weight = one_hot * (1-preds_cp) + (1-one_hot) * preds_cp
                weight = weight ** gamma

                adv_preds_cp = softmax(adv_preds.detach().cpu())
                adv_weight = one_hot * (1-adv_preds_cp) + (1-one_hot) * adv_preds_cp
                adv_weight = adv_weight ** gamma

                weight = torch.mean(adv_weight)
                loss = loss * weight
            elif self._adv_train == 'Curr':
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
        if self._adv_train == 'Curr':
            adv_acc = measurements['adv_accuracy']
            print ('Current query param', self._attack_params_list[self._query_strength], adv_acc)
            window_size = 2000
            smooth_weight = 0.9
            self._adv_acc_list.append(adv_acc)
            if len(self._adv_acc_list) == 1:
                self._smoothed_list.append(adv_acc)
            else:
                self._smoothed_list.append(smooth_weight * self._adv_acc_list[-2] + (1 - smooth_weight) * adv_acc)

            self._iter_cnt += 1
            if len(self._smoothed_list) > window_size and self._iter_cnt > window_size:
                z = np.polyfit(range(window_size), self._smoothed_list[-window_size:], 1)
                incr_rate = z[0]
                print (self._iter_cnt, incr_rate, z)
                if abs(incr_rate) < 1e-4 or self._iter_cnt > 3 * window_size:
                # judge whether stable
                    if self._query_strength < len(self._attack_params_list) - 1:
                        self._query_strength += 1
                        print ('Update attack param', self._attack_params_list[self._query_strength])
                        self._adversary = self.get_adversary(self._model, attack_params=self._attack_params_list[self._query_strength])
                        # update adversary for query
                        self._iter_cnt = 0
                        # update counter
                    else:
                        print ('Reach max param', self._attack_params_list[self._query_strength])
        '''
        if self._adv_train == 'Curr':
            smooth_weight = 0.9
            window_size = 1000
            # print (self._attack_param)
            # print (measurements)
            adv_acc = measurements['adv_accuracy']
            self._adv_acc_list.append(adv_acc)
            if len(self._adv_acc_list) == 1:
                self._smoothed_list.append(adv_acc)
            else:
                self._smoothed_list.append(smooth_weight * self._adv_acc_list[-2] + (1 - smooth_weight) * adv_acc)

            if len(self._smoothed_list) > window_size:
                # print (np.mean(self._smoothed_list[-window_size/2:]))
                z = np.polyfit(range(window_size), self._smoothed_list[-window_size:], 1)
                incr_rate = z[0]
                print (incr_rate)
                if abs(incr_rate) < 1e-4 and self._iter_cnt > window_size:
                # judge whether stable
                    print (self._attack_params)
                    if self._attack_params[1] < self._max_attack_params[1]:
                        self._attack_params[1] += self._incr_step
                        print ('Update attack param', self._attack_params)
                        self._adversary = self.get_adversary(self._model, attack_params=self._attack_params)
                        self._iter_cnt = 0
                    else:
                        print ('Reach max param', self._max_attack_params)
                else:
                    self._iter_cnt += 1
        '''
        return measurements

    def to(self, device, **kwargs):
        self._device = device
        self._model.to(device, **kwargs)
        if self._embedding_model:
            self._embedding_model.to(device, **kwargs)

    def state_dict(self):
        state = {
            'model_state_dict': self._model.state_dict(),
            'optimizers': [ optimizer.state_dict() for optimizer in self._optimizers ],
        }
        if self._embedding_model:
            state.update(
                {'embedding_model_state_dict':
                    self._embedding_model.state_dict()})

        '''
        if self._adv_train == 'new':
            state.update(
                {'tn_state_dict':
                    self._task_net.state_dict()})
            state.update(
                {'tn_optimizer':
                    self._task_net_optim.state_dict()})
        '''
        return state
