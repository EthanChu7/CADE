import numpy as np
import torch
import torch.nn.functional as F
from models.tabular_model import LinearRegression, MLPRegression
from attacker_cade import CADEObservable
import pandas as pd
from config_cade import get_config_syn
from preprocess.preprocess_tab import get_measurement_data


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    args = get_config_syn()
    print(args)

    substitute = args.substitute
    epsilons = args.epsilons
    num_steps = args.num_steps
    lr = args.lr
    path_lin = args.path_lin
    path_lin_pgd = args.path_lin_pgd
    path_mlp = args.path_mlp
    path_mlp_pgd = args.path_mlp_pgd

    y_index = 3
    feat_train, feat_test, labels_train, labels_test, endo_train, endo_test, causal_dag = get_measurement_data(root='./data/', y_index=y_index)

    ranges = torch.max(endo_test, dim=0).values - torch.min(endo_test, dim=0).values
    print(ranges)


    mses_lin = []
    mses_lin_pgd = []
    mses_mlp = []
    mses_mlp_pgd = []

    # load checkpoint
    ckpt_lin = torch.load(path_lin)
    ckpt_lin_pgd = torch.load(path_lin_pgd)
    ckpt_mlp = torch.load(path_mlp)
    ckpt_mlp_pgd = torch.load(path_mlp_pgd)

    model_lin = LinearRegression(feat_train.shape[1], 1)
    model_lin.load_state_dict(ckpt_lin["model_state"])
    model_lin.eval()
    pred_clean_lin = model_lin(feat_test)
    mse_clean_lin = F.mse_loss(pred_clean_lin.squeeze(), labels_test)
    mses_lin.append([mse_clean_lin.item()] * len(epsilons))

    model_lin_pgd = LinearRegression(feat_train.shape[1], 1)
    model_lin_pgd.load_state_dict(ckpt_lin_pgd["model_state"])
    model_lin_pgd.eval()
    pred_clean_lin_pgd = model_lin_pgd(feat_test)
    mse_clean_lin_pgd = F.mse_loss(pred_clean_lin_pgd.squeeze(), labels_test)
    mses_lin_pgd.append([mse_clean_lin_pgd.item()] * len(epsilons))

    model_mlp = MLPRegression(feat_train.shape[1], 32, 1)
    model_mlp.load_state_dict(ckpt_mlp["model_state"])
    model_mlp.eval()
    pred_clean_mlp = model_mlp(feat_test)
    mse_clean_mlp = F.mse_loss(pred_clean_mlp.squeeze(), labels_test)
    mses_mlp.append([mse_clean_mlp.item()] * len(epsilons))

    model_mlp_pgd = MLPRegression(feat_train.shape[1], 32, 1)
    model_mlp_pgd.load_state_dict(ckpt_mlp_pgd["model_state"])
    model_mlp_pgd.eval()
    pred_clean_mlp_pgd = model_mlp_pgd(feat_test)
    mse_clean_mlp_pgd = F.mse_loss(pred_clean_mlp_pgd.squeeze(), labels_test)
    mses_mlp_pgd.append([mse_clean_mlp_pgd.item()] * len(epsilons))

    # print(mses_lin)
    # print(mses_lin_pgd)
    # print(mses_mlp)
    # print(mses_mlp_pgd)

    l_attacking_nodes = [[4], [4], [5], [5], [5, 6], [5, 6]]  # 4: CP, 5: C1, 6: C2
    l_causal_layer = [True, False, True, False, True, False]
    l_dags = [3, 3, 2, 2, 1, 1]

    if substitute == 'lin':
        model_base = model_lin
    elif substitute == 'mlp':
        model_base = model_mlp

    # attacker = CADEObservable(causal_dag, y_index, model_base)

    for mode in range(len(l_attacking_nodes)):
        attacker = CADEObservable(causal_dag,
                                  attacking_nodes=l_attacking_nodes[mode],
                                  y_index=y_index,
                                  substitute=model_base,
                                  l_dag=l_dags[mode])
        # attacker.attacking_nodes = np.array(l_attacking_nodes[mode])
        sub_ranges = ranges[l_attacking_nodes[mode]]
        print(sub_ranges)
        mses_lin_mode = []
        mses_lin_pgd_mode = []
        mses_mlp_mode = []
        mses_mlp_pgd_mode = []

        for i in range(len(epsilons)):
            print("Attacking node: {}, epsilon: {}, causal_layer: {}".format(attacker.attacking_nodes, epsilons[i], l_causal_layer[mode]))
            x_adv = attacker.attack(endo_test,
                                    epsilon=epsilons[i] * sub_ranges,
                                    causal_layer=l_causal_layer[mode],
                                    num_steps=num_steps,
                                    lr=lr)

            with torch.no_grad():
                pred_adv_lin = model_lin(x_adv)
                mse_adv_lin = F.mse_loss(pred_adv_lin.squeeze(), labels_test)
                mses_lin_mode.append(mse_adv_lin.item())

                pred_adv_lin_pgd = model_lin_pgd(x_adv)
                mse_adv_lin_pgd = F.mse_loss(pred_adv_lin_pgd.squeeze(), labels_test)
                mses_lin_pgd_mode.append(mse_adv_lin_pgd.item())

                pred_adv_mlp = model_mlp(x_adv)
                mse_adv_mlp = F.mse_loss(pred_adv_mlp.squeeze(), labels_test)
                mses_mlp_mode.append(mse_adv_mlp.item())

                pred_adv_mlp_pgd = model_mlp_pgd(x_adv)
                mse_adv_mlp_pgd = F.mse_loss(pred_adv_mlp_pgd.squeeze(), labels_test)
                mses_mlp_pgd_mode.append(mse_adv_mlp_pgd.item())

        mses_lin.append(mses_lin_mode)
        mses_lin_pgd.append(mses_lin_pgd_mode)
        mses_mlp.append(mses_mlp_mode)
        mses_mlp_pgd.append(mses_mlp_pgd_mode)


    print(mses_lin)
    print(mses_lin_pgd)
    print(mses_mlp)
    print(mses_mlp_pgd)

    # save and convert mse to rmse
    df_lin = pd.DataFrame(np.sqrt(np.array(mses_lin)), columns=epsilons)
    df_lin.to_excel('lin_{}.xlsx'.format(substitute))

    df_lin_pgd = pd.DataFrame(np.sqrt(np.array(mses_lin_pgd)), columns=epsilons)
    df_lin_pgd.to_excel('lin_pdg_{}.xlsx'.format(substitute))

    df_mlp = pd.DataFrame(np.sqrt(np.array(mses_mlp)), columns=epsilons)
    df_mlp.to_excel('mlp_{}.xlsx'.format(substitute))

    df_mlp_pgd = pd.DataFrame(np.sqrt(np.array(mses_mlp_pgd)), columns=epsilons)
    df_mlp_pgd.to_excel('mlp_pgd_{}.xlsx'.format(substitute))


if __name__ == '__main__':
    main()






