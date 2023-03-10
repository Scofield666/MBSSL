# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.


import math
import torch
from torch.optim.optimizer import Optimizer
import time
import numpy as np

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MetaBalance(Optimizer):
    r"""Implements MetaBalance algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        relax factor: the hyper-parameter to control the magnitude proximity
        beta: the hyper-parameter to control the moving averages of magnitudes, set as 0.9 empirically

    """

    def __init__(self, params, relax_factor=0.7, beta=0.9):
        if not 0.0 <= relax_factor < 1.0:
            raise ValueError("Invalid relax factor: {}".format(relax_factor))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta: {}".format(beta))
        defaults = dict(relax_factor=relax_factor, beta=beta)
        super(MetaBalance, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, loss_array):  # , closure=None
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # loss = None
        # if closure is not None:
        #     with torch.enable_grad():
        #         loss = closure()

        self.balance_GradMagnitudes(loss_array)

        # return loss

    def balance_GradMagnitudes(self, loss_array):
        for loss_index, loss in enumerate(loss_array):
            loss.backward(retain_graph=True)
            for group in self.param_groups:
                for p in group['params']:

                    if p.grad is None:
                        print("breaking")
                        break

                    if p.grad.is_sparse:
                        raise RuntimeError('MetaBalance does not support sparse gradients')

                    if p.grad.equal(torch.zeros_like(p.grad)):
                        continue

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        for j, _ in enumerate(loss_array):
                            if j == 0:
                                p.norms = [torch.zeros(1).cuda()]
                            else:
                                p.norms.append(torch.zeros(1).cuda())

                    # calculate moving averages of gradient magnitudes
                    beta = group['beta']
                    p.norms[loss_index] = (p.norms[loss_index] * beta) + ((1 - beta) * torch.norm(p.grad))

                    # narrow the magnitude gap between the main gradient and each auxilary gradient
                    relax_factor = group['relax_factor']
                    p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (1.0 - relax_factor)

                    if loss_index == 0:
                        state['sum_gradient'] = torch.zeros_like(p.data)
                        state['sum_gradient'] += p.grad
                    else:
                        state['sum_gradient'] += p.grad

                    # have to empty p.grad, otherwise the gradient will be accumulated
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

                    if loss_index == len(loss_array) - 1:
                        p.grad = state['sum_gradient']


class MetaBalance2(Optimizer):
    r"""Implements MetaBalance algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        relax factor: the hyper-parameter to control the magnitude proximity
        beta: the hyper-parameter to control the moving averages of magnitudes, set as 0.9 empirically

    """

    def __init__(self, params, relax_factor=0.7, beta=0.9):
        if not 0.0 <= relax_factor < 1.0:
            raise ValueError("Invalid relax factor: {}".format(relax_factor))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta: {}".format(beta))
        defaults = dict(relax_factor=relax_factor, beta=beta)
        super(MetaBalance2, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, loss_array, nonshared_idx):  # , closure=None
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # loss = None
        # if closure is not None:
        #     with torch.enable_grad():
        #         loss = closure()
        self.balance_GradMagnitudes(loss_array, nonshared_idx)

        # return loss

    def balance_GradMagnitudes(self, loss_array, nonshared_idx):

        for loss_index, loss in enumerate(loss_array):
            loss.backward(retain_graph=True)
            for group in self.param_groups:
                for p_idx, p in enumerate(group['params']):
                    if p_idx == nonshared_idx:
                        continue

                    if p.grad is None:
                        print("breaking")
                        break

                    if p.grad.is_sparse:
                        raise RuntimeError('MetaBalance does not support sparse gradients')

                    # if p.grad.equal(torch.zeros_like(p.grad)):
                    #     continue

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        for j, _ in enumerate(loss_array):
                            if j == 0:
                                p.norms = [torch.zeros(1).cuda()]
                            else:
                                p.norms.append(torch.zeros(1).cuda())

                    # calculate moving averages of gradient magnitudes
                    beta = group['beta']
                    # ?????????iter????????????norm????????????????????????????????????iter?????????
                    p.norms[loss_index] = (p.norms[loss_index] * beta) + ((1 - beta) * torch.norm(p.grad))

                    # narrow the magnitude gap between the main gradient and each auxilary gradient
                    relax_factor = group['relax_factor']

                    p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                            1.0 - relax_factor)

                    if loss_index == 0:
                        state['sum_gradient'] = torch.zeros_like(p.data)
                        state['sum_gradient'] += p.grad
                    else:
                        state['sum_gradient'] += p.grad

                    # have to empty p.grad, otherwise the gradient will be accumulated
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

                    if loss_index == len(loss_array) - 1:
                        p.grad = state['sum_gradient']


class MetaBalance3(Optimizer):
    r"""Implements MetaBalance algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        relax factor: the hyper-parameter to control the magnitude proximity
        beta: the hyper-parameter to control the moving averages of magnitudes, set as 0.9 empirically

    """

    def __init__(self, params, relax_factor=0.7, beta=0.9):
        if not 0.0 <= relax_factor < 1.0:
            raise ValueError("Invalid relax factor: {}".format(relax_factor))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta: {}".format(beta))
        defaults = dict(relax_factor=relax_factor, beta=beta)
        super(MetaBalance3, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, loss_array, nonshared_idx, meta_strategy):  # , closure=None
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # loss = None
        # if closure is not None:
        #     with torch.enable_grad():
        #         loss = closure()
        self.balance_GradMagnitudes2(loss_array, nonshared_idx, meta_strategy)

        # return loss
    def balance_GradMagnitudes(self, loss_array, nonshared_idx, meta_strategy):
        grad_task = []  # ???????????????????????????????????????
        for loss_index, loss in enumerate(loss_array):
            loss.backward(retain_graph=True)
            for group in self.param_groups:
                for p_idx, p in enumerate(group['params']):

                    if p_idx == nonshared_idx:
                        continue

                    if p.grad is None:
                        print("breaking")
                        break

                    if p.grad.is_sparse:
                        raise RuntimeError('MetaBalance does not support sparse gradients')

                    # if p.grad.equal(torch.zeros_like(p.grad)):
                    #     continue

                    state = self.state[p]  # ????????????p???????????????loss????????????????????????????????????

                    # State initialization
                    if len(state) == 0:
                        for j, _ in enumerate(loss_array):
                            if j == 0:
                                p.norms = [torch.zeros(1).cuda()]
                            else:
                                p.norms.append(torch.zeros(1).cuda())

                    # calculate moving averages of gradient magnitudes
                    beta = group['beta']
                    # ?????????iter????????????norm????????????????????????????????????iter?????????
                    p.norms[loss_index] = (p.norms[loss_index] * beta) + ((1 - beta) * torch.norm(p.grad))

                    # narrow the magnitude gap between the main gradient and each auxilary gradient
                    relax_factor = group['relax_factor']

                    #  strategy A??????????????????????????????????????????????????????????????????
                    if meta_strategy == 1:
                        # ???????????????????????????????????????????????????????????????
                        if p.norms[loss_index] > p.norms[0]:
                            p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                                    1.0 - relax_factor)
                    # strategy C
                    elif meta_strategy == 2:
                        p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                                1.0 - relax_factor)

                    if loss_index == 0:
                        state['sum_gradient'] = torch.zeros_like(p.data)
                        state['sum_gradient'] += p.grad
                    else:
                        state['sum_gradient'] += p.grad

                    # have to empty p.grad, otherwise the gradient will be accumulated
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

                    if loss_index == len(loss_array) - 1:
                        p.grad = state['sum_gradient']

    def balance_GradMagnitudes2(self, loss_array, nonshared_idx, meta_strategy):
        grad_task = []  # ???????????????????????????????????????
        for loss_index, loss in enumerate(loss_array):
            loss.backward(retain_graph=True)
            for group in self.param_groups:
                for p_idx, p in enumerate(group['params']):
                    if loss_index == 0:
                        grad_task.append(p.grad.detach().clone())  # ???????????????????????????????????????

                    if p_idx == nonshared_idx:
                        continue

                    if p.grad is None:
                        print("breaking")
                        break

                    if p.grad.is_sparse:
                        raise RuntimeError('MetaBalance does not support sparse gradients')

                    # if p.grad.equal(torch.zeros_like(p.grad)):
                    #     continue

                    state = self.state[p]  # ????????????p???????????????loss????????????????????????????????????

                    # State initialization
                    if len(state) == 0:
                        for j, _ in enumerate(loss_array):
                            if j == 0:
                                p.norms = [torch.zeros(1).cuda()]
                            else:
                                p.norms.append(torch.zeros(1).cuda())

                    # calculate moving averages of gradient magnitudes
                    beta = group['beta']
                    # ?????????iter????????????norm????????????????????????????????????iter?????????
                    p.norms[loss_index] = (p.norms[loss_index] * beta) + ((1 - beta) * torch.norm(p.grad))

                    # narrow the magnitude gap between the main gradient and each auxilary gradient
                    relax_factor = group['relax_factor']

                    # strategy 0: r * G_aux * |G_tar| / |G_aux|
                    if meta_strategy == 0:
                        # ???????????????????????????????????????????????????
                        if p.norms[loss_index] > p.norms[0]:
                            p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor
                            inner_p = torch.sum(p.grad * grad_task[p_idx])
                            if inner_p < 0:
                                p.grad = p.grad - inner_p / (torch.norm(grad_task[p_idx]) ** 2) * grad_task[p_idx]

                    #  strategy A??????????????????????????????????????????????????????????????????
                    if meta_strategy == 1:
                        # ???????????????????????????????????????????????????????????????
                        if p.norms[loss_index] > p.norms[0]:
                            p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                                    1.0 - relax_factor)
                    # strategy C
                    elif meta_strategy == 2:
                        p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                                1.0 - relax_factor)
                    # ?????????????????????????????????????????????
                    elif meta_strategy == 3:
                        p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                                1.0 - relax_factor)

                        inner_p = torch.sum(p.grad * grad_task[p_idx])
                        if inner_p < 0:
                            # print('Loss: {}, Para: {}, start adjusting direction'.format(loss_index, p_idx))
                            p.grad = p.grad - inner_p / (torch.norm(grad_task[p_idx]) ** 2) * grad_task[p_idx]

                    # ???????????????????????????????????????????????????????????????
                    elif meta_strategy == 4:
                        if p.norms[loss_index] > p.norms[0]:
                            p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                                    1.0 - relax_factor)
                        else:
                            inner_p = torch.sum(p.grad * grad_task[p_idx])
                            if inner_p < 0:
                                # print('Loss: {}, Para: {}, start adjusting direction'.format(loss_index, p_idx))
                                p.grad = p.grad - inner_p / (torch.norm(grad_task[p_idx]) ** 2) * grad_task[p_idx]
                    # ????????????????????????????????????????????????meta balance????????????pcgrad
                    elif meta_strategy == 5:
                        inner_p = torch.sum(p.grad * grad_task[p_idx])
                        if p.norms[loss_index] > p.norms[0] or inner_p > 0:
                            p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                                    1.0 - relax_factor)
                        else:
                            p.grad = p.grad - inner_p / (torch.norm(grad_task[p_idx]) ** 2) * grad_task[p_idx]

                    elif meta_strategy == 6:
                        inner_p = torch.sum(p.grad * grad_task[p_idx])
                        # 1. ????????????
                        if p.norms[loss_index] > p.norms[0]:
                            # 1.1 ???????????????????????????
                            if inner_p >= 0:
                                p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                                        1.0 - relax_factor)
                            # 1.2 ???????????????????????????
                            else:
                                p.grad = p.grad - inner_p / (torch.norm(grad_task[p_idx]) ** 2) * grad_task[p_idx]
                        # 2. ????????????
                        else:
                            # 2.1 ???????????????????????????
                            if inner_p >= 0:
                                p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                                        1.0 - relax_factor)
                            # 2.2 ???????????????????????????
                            else:
                                p.grad = p.grad - inner_p / (torch.norm(grad_task[p_idx]) ** 2) * grad_task[p_idx]


                    elif meta_strategy == 7:
                        # ?????????????????????????????????????????????????????????????????????????????????
                        inner_p = torch.sum(p.grad * grad_task[p_idx])
                        if p.norms[loss_index] > p.norms[0]:
                            if inner_p >= 0:
                                p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                                        1.0 - relax_factor)
                            else:
                                p.grad = p.grad - inner_p / (torch.norm(grad_task[p_idx]) ** 2) * grad_task[p_idx]

                    elif meta_strategy == 8:
                        # ?????????????????????????????????????????????????????????
                        if p.norms[loss_index] > p.norms[0]:
                            inner_p = torch.sum(p.grad * grad_task[p_idx])
                            if inner_p < 0:
                                p.grad = p.grad - inner_p / (torch.norm(grad_task[p_idx]) ** 2) * grad_task[p_idx]
                            p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                                    1.0 - relax_factor)
                    elif meta_strategy == 9:
                        # ?????????????????????????????????????????????????????????
                        if p.norms[loss_index] > p.norms[0]:
                            p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                                    1.0 - relax_factor)
                            inner_p = torch.sum(p.grad * grad_task[p_idx])
                            if inner_p < 0:
                                p.grad = p.grad - inner_p / (torch.norm(grad_task[p_idx]) ** 2) * grad_task[p_idx]

                    elif meta_strategy == 10:
                        # ????????????????????????????????????????????????????????????????????????????????????????????????????????????norm
                        inner_p = torch.sum(p.grad * grad_task[p_idx])
                        if p.norms[loss_index] > p.norms[0]:
                            if inner_p < 0:
                                p.grad = p.grad - inner_p / (torch.norm(grad_task[p_idx]) ** 2) * grad_task[p_idx]
                            # ?????????????????????????????????????????????????????????
                            new_norm = (p.norms[loss_index] * beta) + ((1 - beta) * torch.norm(p.grad))
                            if new_norm > p.norms[0]:
                                p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                                        1.0 - relax_factor)

                    elif meta_strategy == 11:
                        # ?????????pcgrad
                        inner_p = torch.sum(p.grad * grad_task[p_idx])
                        if inner_p < 0:
                            p.grad = p.grad - inner_p / (torch.norm(grad_task[p_idx]) ** 2) * grad_task[p_idx]

                    elif meta_strategy == 12:
                        # ??????metabalance??????????????????pcgrad
                        p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                                1.0 - relax_factor)

                        inner_p = torch.sum(p.grad * grad_task[p_idx])
                        if inner_p < 0:
                            p.grad = p.grad - inner_p / (torch.norm(grad_task[p_idx]) ** 2) * grad_task[p_idx]

                    elif meta_strategy == 13:
                        # ??????pcgrad??????????????????meta balance
                        inner_p = torch.sum(p.grad * grad_task[p_idx])
                        if inner_p < 0:
                            p.grad = p.grad - inner_p / (torch.norm(grad_task[p_idx]) ** 2) * grad_task[p_idx]

                        p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                                1.0 - relax_factor)

                    if loss_index == 0:
                        state['sum_gradient'] = torch.zeros_like(p.data)
                        state['sum_gradient'] += p.grad
                    else:
                        state['sum_gradient'] += p.grad

                    # have to empty p.grad, otherwise the gradient will be accumulated
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

                    if loss_index == len(loss_array) - 1:
                        p.grad = state['sum_gradient']


class MetaBalance4(Optimizer):
    r"""Implements MetaBalance algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        relax factor: the hyper-parameter to control the magnitude proximity
        beta: the hyper-parameter to control the moving averages of magnitudes, set as 0.9 empirically

    """

    def __init__(self, params, relax_factor=0.7, beta=0.9):
        if not 0.0 <= relax_factor < 1.0:
            raise ValueError("Invalid relax factor: {}".format(relax_factor))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta: {}".format(beta))
        defaults = dict(relax_factor=relax_factor, beta=beta)
        super(MetaBalance4, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, loss_array, nonshared_idx, meta_strategy, epoch_grad_dict, idx2name_dict):  # , closure=None
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # loss = None
        # if closure is not None:
        #     with torch.enable_grad():
        #         loss = closure()
        self.balance_GradMagnitudes(loss_array, nonshared_idx, meta_strategy, epoch_grad_dict, idx2name_dict)

        # return loss

    def balance_GradMagnitudes(self, loss_array, nonshared_idx, meta_strategy, epoch_grad_dict, idx2name_dict):

        for loss_index, loss in enumerate(loss_array):
            loss.backward(retain_graph=True)
            for group in self.param_groups:
                for p_idx, p in enumerate(group['params']):
                    if p_idx == nonshared_idx:
                        continue

                    if p.grad is None:
                        print("breaking")
                        break

                    if p.grad.is_sparse:
                        raise RuntimeError('MetaBalance does not support sparse gradients')

                    # if p.grad.equal(torch.zeros_like(p.grad)):
                    #     continue

                    state = self.state[p]  # ????????????p???????????????loss????????????????????????????????????

                    # State initialization
                    if len(state) == 0:
                        for j, _ in enumerate(loss_array):
                            if j == 0:
                                p.norms = [torch.zeros(1).cuda()]
                            else:
                                p.norms.append(torch.zeros(1).cuda())

                    # calculate moving averages of gradient magnitudes
                    beta = group['beta']
                    # ?????????iter????????????norm????????????????????????????????????iter?????????
                    p.norms[loss_index] = (p.norms[loss_index] * beta) + ((1 - beta) * torch.norm(p.grad))

                    # narrow the magnitude gap between the main gradient and each auxilary gradient
                    relax_factor = group['relax_factor']

                    # s1???
                    if meta_strategy == 1:
                        # ???????????????????????????????????????????????????????????????
                        if p.norms[loss_index] > p.norms[0]:
                            p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                                    1.0 - relax_factor)
                    # s2: metabalance
                    elif meta_strategy == 2:
                        p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                                1.0 - relax_factor)

                    p_name = idx2name_dict[p_idx]
                    epoch_grad_dict[p_name][loss_index] += torch.norm(p.grad)

                    if loss_index == 0:
                        state['sum_gradient'] = torch.zeros_like(p.data)
                        state['sum_gradient'] += p.grad
                    else:
                        state['sum_gradient'] += p.grad

                    # have to empty p.grad, otherwise the gradient will be accumulated
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

                    if loss_index == len(loss_array) - 1:
                        p.grad = state['sum_gradient']