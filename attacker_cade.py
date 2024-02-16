import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class CADELatent:
    def __init__(self, generative_model, attacking_nodes, substitute, target_idx=0, l_dag=1, num_causal_variables=4, device='cuda'):
        """

        @param generative_model: The generative model that can infer the latent variables and generate images from latent
        @param attacking_nodes: The indices of variables we aim to attack at
        @param substitute: The substitute model we use to perform white-box attack
        @param target_idx: The index of y in z
        @param l_dag: the depth of causal graph
        @param num_causal_variables: the number of causal_variables
        @param device: 'cuda' or 'cpu'
        """
        self.generative_model = generative_model
        self.generative_model.eval()
        self.generative_model.requires_grad_(False)

        self.substitute = substitute
        self.substitute.eval()
        self.substitute.requires_grad_(False)
        self.attacking_nodes = attacking_nodes
        self.target_idx = target_idx
        self.l_dag = l_dag
        self.num_causal_variables = num_causal_variables

        self.device = device

    def attack_whitebox(self, x, label, epochs=20, lr=0.5, type_loss='pendulum', epsilon=2., causal_layer=True):
        """

        @param x: the original examples with shape [n,.,.,.] where n denotes the batch_size
        @param label: the label with shape [n, ]
        @param epochs: num_steps
        @param lr: step_size
        @param type_loss: the loss type
        @param epsilon: the budget
        @param causal_layer: apply causal layer or not
        @return: the adversarial examples
        """
        self.generative_model.requires_grad_(False)
        self.generative_model.eval()

        full_z = self.generative_model(x)
        full_z_adv = full_z.clone()
        full_z_adv.requires_grad = False
        attack_z_adv = full_z_adv[:, self.attacking_nodes]
        attack_z_adv.requires_grad = True
        optimizer = torch.optim.Adam([attack_z_adv], lr=lr)

        min_clip = torch.zeros_like(full_z) - np.inf
        max_clip = torch.zeros_like(full_z) + np.inf
        min_clip[:, self.attacking_nodes] = -epsilon
        max_clip[:, self.attacking_nodes] = epsilon
        # print(max_clip)

        # only apply in the experiment on Pendulum
        # Abduction: recover the exogenous
        # And set the binary mask m
        if causal_layer:  # the abduction step to recover the exogenous
            causal_z = full_z[:, :self.num_causal_variables]  # 4 causal variables in Pendulum
            causal_z_nlr = self.generative_model.prior.enc_nlr(causal_z)  # f(z)
            exogenous = self.generative_model.prior.get_eps(causal_z_nlr)  # (I-A^T)f(z), in tenser form, f(z)@(I-A)

            mask_is_intervened = torch.zeros_like(exogenous)
            mask_is_intervened[:, self.attacking_nodes] = 1.  # set the attacking variables to 1

        for epoch in range(epochs):
            optimizer.zero_grad()

            full_z_adv[:, self.attacking_nodes] = attack_z_adv
            diff_full_z = full_z_adv - full_z
            diff_full_z = torch.clamp(diff_full_z, min_clip, max_clip)  # clamp, norm(z)_{inf} <= eps
            full_z_adv = full_z + diff_full_z
            # print(full_z[0])
            # print(full_z_adv[0])

            # only apply in the experiment on Pendulum
            if causal_layer:
                causal_z_adv = full_z_adv[:, :self.num_causal_variables]  # 4 causal variables in Pendulum
                other_z_adv = full_z_adv[:, self.num_causal_variables:]
                causal_z_nlr_adv = self.generative_model.prior.enc_nlr(causal_z_adv)  # f(z')

                # propagate the causal effect l_dag times, where l_dag denotes the depth of the casual DAG
                for _ in range(self.l_dag):
                    causal_z_nlr_adv = (1 - mask_is_intervened) * (causal_z_nlr_adv @ self.generative_model.prior.A) + mask_is_intervened * causal_z_nlr_adv + (1 - mask_is_intervened) * exogenous

                causal_z_adv = self.generative_model.prior.prior_nlr(causal_z_nlr_adv)  # f^{-1}(.)
                full_z_adv = torch.cat([causal_z_adv, other_z_adv], dim=1)

            # print(full_z_adv[0])
            x_adv = self.generative_model.decoder(full_z_adv)
            out = self.substitute(x_adv)
            # loss
            if type_loss == 'celeba':
                target_label = torch.abs(1-label)  # for CelebA dataset
                # loss_pred = F.cross_entropy(out, target_label) - F.cross_entropy(out, label)
                loss_pred = 0
                N = out.shape[0]
                # out = F.softmax(out, dim=1)
                for i in range(N):
                    loss_pred -= out[i, target_label[i]]
                    loss_pred += out[i, label[i]]
                loss_pred = loss_pred / N  # a variant of f2 formulation in C&W attack
            elif type_loss == 'pendulum':
                loss_pred = -F.cross_entropy(out, label)
            else:
                loss_pred = -F.cross_entropy(out, label)
            # loss_size = torch.mean(torch.sum(torch.abs(full_z_adv - full_z), dim=1))
            # # loss = loss_pred + alpha * loss_size
            loss = loss_pred

            loss.backward(retain_graph=True)
            optimizer.step()
            # print("epoch: {}, loss_pred: {}, loss_size: {}".format(epoch, loss_pred, loss_size))
            print("epoch: {}, loss_pred: {}".format(epoch, loss_pred))

        return x_adv

    def attack_random(self, x, epsilon=2., causal_layer=True):
        """

        @param x: the original examples with shape [n,.,.,.] where n denotes the batch_size
        @param epsilon: the budget
        @param causal_layer: apply causal layer or not
        @return: the adversarial examples
        """
        self.generative_model.requires_grad_(False)
        self.generative_model.eval()

        full_z = self.generative_model(x)
        full_z_adv = full_z.clone()
        attack_z_adv = (torch.rand((full_z.shape[0], len(self.attacking_nodes))).to(self.device) - 0.5) * 2 * epsilon
        full_z_adv[:, self.attacking_nodes] += attack_z_adv

        if causal_layer:
            causal_z = full_z[:, :self.num_causal_variables]  # 4 causal variables in Pendulum
            causal_z_nlr = self.generative_model.prior.enc_nlr(causal_z)  # f(z)
            exogenous = self.generative_model.prior.get_eps(causal_z_nlr)  # (I-A^T)f(z), in tenser form, f(z)@(I-A)

            mask_is_intervened = torch.zeros_like(exogenous)
            mask_is_intervened[:, self.attacking_nodes] = 1.

            causal_z_adv = full_z_adv[:, :self.num_causal_variables]  # 4 causal variables in Pendulum
            other_z_adv = full_z_adv[:, self.num_causal_variables:]
            causal_z_nlr_adv = self.generative_model.prior.enc_nlr(causal_z_adv)  # f(z')
            # propagate the causal effect l_dag times, where l_dag denotes the depth of the casual DAG
            for _ in range(self.l_dag):
                causal_z_nlr_adv = (1 - mask_is_intervened) * (causal_z_nlr_adv @ self.generative_model.prior.A) + mask_is_intervened * causal_z_nlr_adv + (1 - mask_is_intervened) * exogenous

            causal_z_adv = self.generative_model.prior.prior_nlr(causal_z_nlr_adv)  # f^{-1}(.)
            full_z_adv = torch.cat([causal_z_adv, other_z_adv], dim=1)

        x_adv = self.generative_model.decoder(full_z_adv)

        return x_adv


class CADEObservable:
    def __init__(self, causal_dag, attacking_nodes, y_index, substitute):
        """

        :param causal_dag: the weighted causal DAG where the data-generating process: X = XA + E
        :param y_index: the index of y in X
        :param model_reg: the regression model we are about to attack
        """
        self.causal_dag = causal_dag
        self.attacking_nodes = attacking_nodes
        self.y_index = y_index
        print(self.attacking_nodes)

        self.substitute = substitute
        self.substitute.requires_grad_(False)

    def markov_blanket(self, causal_dag, y_index):
        children = np.nonzero(causal_dag[y_index])[0]
        parents = np.nonzero(causal_dag[:, y_index])[0]
        co_parents = set()
        for child in children:
            sub_co_parents = np.nonzero(causal_dag[:, child])[0]
            for sub_co_parent in sub_co_parents:
                if sub_co_parent != y_index:
                    co_parents.add(sub_co_parent)
        co_parents = np.array(list(co_parents))
        print('children: {}, parent: {}, co_parents: {}'.format(children, parents, co_parents))

        return children, parents, co_parents

    def recover_exogenous_linear(self, causal_dag, endogenous):
        """
        Parameters
        ----------
        causal_dag: torch.FloatTensor of shape [d, d]
        endogenous: torch.FloatTensor of shape [n, d]

        Returns
        -------
        torch.FloatTensor of shape [n, d]
        """
        # u = (I-A^T)x
        # In tensor form, x = x @ A + e -> e = x @ (I - A)
        identity_matrix = torch.eye(causal_dag.shape[0])
        exogenous = endogenous @ (identity_matrix - causal_dag)
        return exogenous

    def attack(self, endogenous, epsilon=1., causal_layer=True, num_steps=150, step_size=0.1):
        exogenous = self.recover_exogenous_linear(self.causal_dag, endogenous)
        full_endogenous = endogenous.clone()
        full_endogenous.requires_grad = False
        attacking_endogenous = endogenous[:, self.attacking_nodes]
        attacking_endogenous.requires_grad = True

        optimizer = torch.optim.Adam([attacking_endogenous], lr=step_size)

        for epoch in range(num_steps):
            optimizer.zero_grad()

            previous_endogenous = full_endogenous.clone()

            mask_is_intervened = torch.zeros_like(exogenous)
            min_clip = torch.zeros_like(full_endogenous) - np.inf
            max_clip = torch.zeros_like(full_endogenous) + np.inf
            mask_is_intervened[:, self.attacking_nodes] = 1.
            min_clip[:, self.attacking_nodes] = -epsilon
            max_clip[:, self.attacking_nodes] = epsilon

            for i in range(len(self.attacking_nodes)):
                full_endogenous[:, self.attacking_nodes[i]] = attacking_endogenous[:, i]  # concat the column

            diff_endo = full_endogenous - endogenous
            diff_endo = torch.clamp(diff_endo, min_clip, max_clip)

            full_endogenous = endogenous + diff_endo


            if causal_layer:
                for _ in range(3): # the depth of causal graph is 3
                    full_endogenous = (1-mask_is_intervened) * (full_endogenous @ self.causal_dag) + mask_is_intervened * full_endogenous + (1-mask_is_intervened) * exogenous

            # feed the intervened endogenous to surrogate model to get outcome
            x_adv = torch.cat((full_endogenous[:, :self.y_index], full_endogenous[:, self.y_index+1:]), dim=1)
            out = self.substitute(x_adv)

            # loss
            loss_pred = -F.mse_loss(out.squeeze(), endogenous[:, self.y_index])
            loss = loss_pred

            if epoch % 10 == 0:
                print("epoch: {}, loss_ce: {}, loss: {}".format(epoch, loss_pred, loss))

            loss.backward(retain_graph=True)
            optimizer.step()

        return x_adv

    def attack_random(self, endogenous, epsilon=1., causal_layer=True):
        exogenous = self.recover_exogenous_linear(self.causal_dag, endogenous)
        full_endogenous = endogenous.clone()
        attacking_endogenous = endogenous[:, self.attacking_nodes]
        delta_attacking_endogenous = (torch.rand(attacking_endogenous.shape) - 0.5) * 2 * epsilon
        attacking_endogenous += delta_attacking_endogenous

        mask_is_intervened = torch.zeros_like(exogenous)
        mask_is_intervened[:, self.attacking_nodes] = 1.

        for i in range(len(self.attacking_nodes)):
            full_endogenous[:, self.attacking_nodes[i]] = attacking_endogenous[:, i]  # concat the column

        if causal_layer:
            for _ in range(3): # the depth of causal graph is 3
                full_endogenous = (1-mask_is_intervened) * (full_endogenous @ self.causal_dag) + mask_is_intervened * full_endogenous + (1-mask_is_intervened) * exogenous
        # feed the intervened endogenous to surrogate model to get outcome
        x_adv = torch.cat((full_endogenous[:, :self.y_index], full_endogenous[:, self.y_index+1:]), dim=1)

        return x_adv


