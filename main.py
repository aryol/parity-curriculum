from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import random
import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
import token_transformer
from utilities import create_test_matrix_11, calculate_fourier_coefficients
from examples import tasks
import time

import models

def generate_fresh_samples(batch_size, dimension, p, rho):
    if rho == 1:
        return 1 - 2 * torch.bernoulli(torch.ones((batch_size, dimension), device=device) * p)
    else:
        cluster = torch.bernoulli(torch.ones((batch_size, 1), device=device) * rho)
        return 1 - 2 * torch.bernoulli((torch.ones((batch_size, dimension), device=device) * 0.5 - cluster * (0.5 - p)))

    


def build_model(arch, dimension):
    """
    This function creates the model based on the argument given in the command line.
    """
    if arch == 'mlp':
        model = models.MLP(input_dimension=dimension)
    elif arch == 'mup':
        model = models.MaximalUpdate(input_dimension=dimension, width=2**12, depth=6)
    elif arch == 'ntk':
        model = models.NTK(input_dimension=dimension, width=2**13, depth=3)
    elif arch == 'meanfield':
        model = models.MeanField(input_dimension=dimension)
    elif arch == 'transformer':
        model = token_transformer.TokenTransformer(
                seq_len=dimension, output_dim=1, dim=256, depth=6, heads=6, mlp_dim=256)
    return model.to(device)



def train(train_X, train_y, valid_X, valid_y, test_X, test_y, computation_interval=0, verbose_interval=0, monomials=None, print_coefficients=False, model=None, eps=0.001):
    """
    This is the main training function which receives the datasets and does the training (curriculum or normal)
    :param monomials: This argument recieves a mask which shows coefficient of which monomials must be computed. 
    :param curr: This argument is used to activate curriculum learning. If none then normal training. If not none, it is equal to (leap, threshold) of the degree-curriculum algorithm.  
    :param computation_interval: Denotes frequency of computation of valid/test losses and also coefficients of the monomials.
    :return: The function returns epoch_logs (just epochs that computations are done), train_losses, valid_losses, test_losses, coefficients (of monomials denoted by monomials argument during the training), coefficients_norms (used for calculating degree profile), iter_counter (number of iterations done during the optimizaiton).
    
    Note that the test dataset is used for the computation of coefficients of the monomials. 
    """
    if model is None:
        print("Model created.")
        model = build_model(task_params['model'], dimension)
    # Logging arrays
    iter_logs = []
    train_losses = []
    valid_losses = []
    test_losses = []
    train_accs = []
    valid_accs = []
    test_accs = []
    coefficients = []

    # Preparing the dataset
    ## Reshaping
    if train_X is not None:
        train_y = train_y.reshape(-1, 1)
    valid_y = valid_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    ## Creating pytorch tensors
    if train_X is not None:
        train_X = torch.tensor(train_X, device=device)
        train_y = torch.tensor(train_y, device=device)
    valid_X = torch.tensor(valid_X, device=device)
    valid_y = torch.tensor(valid_y, device=device)
    test_X = torch.tensor(test_X, device=device)
    test_y = torch.tensor(test_y, device=device)

    if train_X is not None:
        train_ds = TensorDataset(train_X, train_y)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_ds = TensorDataset(valid_X, valid_y)
    valid_dl = DataLoader(valid_ds, batch_size=test_batch_size)
    test_ds = TensorDataset(test_X, test_y)
    test_dl = DataLoader(test_ds, batch_size=test_batch_size)

    # Defining the optimizer
    if task_params['opt'].lower() == 'sgd':
        print("Using SGD")
        opt = optim.SGD(model.parameters(), lr=task_params['lr'], momentum=momentum, weight_decay=0.0)
    else:
        print("Using Adam")
        opt = optim.Adam(model.parameters(), lr=task_params['lr'])
    
    def hinge_loss(output, target):
        return torch.max(torch.tensor(0), 1 - output * target).mean()
    
    if train_X is not None:
        train_mean = train_y.mean()
    else:
        train_mean = (1 - 2 * task_params['p']) ** 7 # ONLY working for parity 7
        if train_y == 1:
            train_mean *= task_params['rho']
        print("Train_y_mean:", train_mean)

    loss_func = nn.MSELoss()
    if task_params['loss'].lower() == 'hinge':
        print("Using hinge loss.")
        loss_func = hinge_loss

    if task_params['loss'].lower() == 'cov':
        def cov_loss(output, target):
            return torch.max(torch.tensor(0), (target - train_mean) * (target - output)).mean()
        print("Using cov loss.")
        loss_func = cov_loss
    
    
    # Function used for evaluation of the model, i.e., calculation of coefficients and valid/test losses. 
    def model_evaluation(iter, train_loss, train_acc):
        model.eval()
        with torch.no_grad():
            # Computing coefficients of the monomials and the average norm per degree
            if monomials is not None:
                y_pred = torch.vstack([model(xb) for xb, _ in test_dl])
                coefficients.append(calculate_fourier_coefficients(monomials, test_X.cpu().detach().numpy(),
                                                               y_pred.cpu().detach().numpy()))
                # coefficients_norms.append([((coefficients[-1][monomials.sum(axis=1) == dim]) ** 2).sum() for dim in range(dimension + 1)])                           
            # Computing loss on the validation and test data
            valid_loss = 0
            valid_acc = 0
            for xb, yb in valid_dl:
                pred = model(xb)
                valid_loss += loss_func(pred, yb)
                valid_acc += ((pred.sign() * yb) + 1).sum() / 2
            valid_loss /= len(valid_dl)
            valid_acc /= len(valid_y)

            test_loss = 0
            test_acc = 0
            for xb, yb in test_dl:
                pred = model(xb)
                test_loss += loss_func(pred, yb)
                test_acc += ((pred.sign() * yb) + 1).sum() / 2
            test_loss /= len(test_dl) 
            test_acc /= len(test_y)

            if train_loss is None:
                train_loss = valid_loss
                train_acc = valid_acc

            train_loss = train_loss.cpu().detach().numpy()
            valid_loss = valid_loss.cpu().detach().numpy()
            test_loss = test_loss.cpu().detach().numpy()
            train_acc = train_acc.cpu().detach().numpy()
            valid_acc = valid_acc.cpu().detach().numpy()
            test_acc = test_acc.cpu().detach().numpy()

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)
            test_accs.append(test_acc)
            iter_logs.append(iter)

            if (iter % verbose_interval == 0) or (train_loss < eps):
                if (monomials is not None) and print_coefficients:
                    print("Coefficients:", coefficients[-1])
                print(f"Iter: {iter:8}, Train Loss: {train_loss:0.6}, Valid Loss: {valid_loss:0.6}, Test Loss: {test_loss:0.6}, Train Acc: {train_acc:0.3}, Valid Acc: {valid_acc:0.3}, Test Acc: {test_acc:0.3}, Elapsed Time:", time.strftime("%H:%M:%S",time.gmtime(time.time() - start_time)))

    
    # Model's evaluation before training
    model_evaluation(0, None, None)
    iter_counter = 0
    train_loss = torch.tensor(0.0, device=device)
    train_acc = 0
    model.train()
    training_flag = True
    while training_flag:
        if train_X is not None:
            for xb, yb in train_dl:
                pred = model(xb)
                loss = loss_func(pred, yb)
                train_loss += loss
                train_acc += ((pred.sign() * yb) + 1).mean() / 2
                loss.backward()
                opt.step()
                opt.zero_grad()
                iter_counter += 1
                if iter_counter % computation_interval == 0:
                    train_loss /= computation_interval
                    train_acc /= computation_interval
                    model_evaluation(iter_counter, train_loss, train_acc)
                    if train_loss < eps:
                        training_flag = False
                        break
                    model.train()
                    train_loss *= 0
                    train_acc *= 0
        else:
            xb = generate_fresh_samples(batch_size, dimension, p=task_params['p'], rho=task_params['rho'] if train_y==1 else 1.0)
            yb = task_params['target_function'](xb).reshape(-1, 1)
            pred = model(xb)
            loss = loss_func(pred, yb)
            train_loss += loss
            train_acc += ((pred.sign() * yb) + 1).mean() / 2
            loss.backward()
            opt.step()
            opt.zero_grad()
            iter_counter += 1
            if iter_counter % computation_interval == 0:
                train_loss /= computation_interval
                train_acc /= computation_interval
                model_evaluation(iter_counter, train_loss, train_acc)
                if train_loss < eps:
                    training_flag = False
                    break
                model.train()
                train_loss *= 0
                train_acc *= 0

            
    return iter_logs, train_losses, valid_losses, test_losses, train_accs, valid_accs, test_accs, coefficients, model



if __name__ == '__main__':
    
    parser = ArgumentParser(description="Training script for neural networks on different functions",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    # Required runtime params
    parser.add_argument('-task', required=True, type=str, help='name of the task')
    parser.add_argument('-model', required=True, type=str, help='name of the model')
    # parser.add_argument('-epochs', required=True, type=int, help='number of epochs')
    parser.add_argument('-lr', required=True, type=float, help='learning rate')
    parser.add_argument('-seed', required=True, type=int, help='random seed')
    parser.add_argument('-p', required=True, type=float, help='biased distribution')
    parser.add_argument('-rho', required=True, type=float, help='relative size of the biased part')
    parser.add_argument('-curr', type=int, help='using two step curriculum or not')
    parser.add_argument('-train-size', required=True, type=int, help='the size of the biased distribution')
    # Other runtime params
    parser.add_argument('-cuda', required=False, type=str, default='0', help='number of the gpu')
    parser.add_argument('-eps', required=False, type=str, default=0.001, help='threshold to stop')
    parser.add_argument('-loss', required=False, type=str, default="", help='loss function used for training -- default is l2 while hinge and covariance loss can also be selected.')
    parser.add_argument('-opt', default='sgd', type=str, help='sgd or adam')
    parser.add_argument('-batch-size', default=64, type=int, help='batch size')
    parser.add_argument('-test-batch-size', type=int, default=8192, help='batch size for test samples')
    parser.add_argument('-verbose-int', default=1, type=int, help="the interval between prints")
    parser.add_argument('-compute-int', default=1, type=int, help="the interval between computations of monomials and losses")
    
    args = parser.parse_args()
    start_time = time.time()
    # General setup of the experiments
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    momentum = 0.0

    if args.task not in tasks:
        print("Task not found.")
        exit()
    task_params = tasks[args.task]
    dimension = task_params['dimension']
    task_params.update(vars(args))
    batch_size = task_params['batch_size']
    test_batch_size = task_params['test_batch_size']
    mask = task_params['mask']
    curriculum = bool(task_params['curr'])
    
    if mask.shape[1] < dimension:
        mask = np.hstack((mask, np.zeros((mask.shape[0], dimension - mask.shape[1]), dtype=int)))

    print(vars(args))

    # Setting the seeds
    np.random.seed(task_params['seed'])
    random.seed(task_params['seed'])
    torch.manual_seed(task_params['seed'])

    

    if task_params['train_size'] > 0:
        # Generating train, valid, and test data. We use num_samples = 0 as an indication to create the whole space. 
        train_X_biased = create_test_matrix_11(int(task_params['train_size'] * task_params['rho']), dimension, p=task_params['p'])
        train_X_uniform = create_test_matrix_11(int(task_params['train_size'] * (1 - task_params['rho'])), dimension)
        train_X = np.vstack([train_X_biased, train_X_uniform])
        train_y = task_params['target_function'](train_X)

        # We do not know the easy samples a priori. We use the following rule to find them:
        train_X_biased = train_X[train_X.mean(axis = 1) > 0.5 - task_params['p']]
        train_y_biased = train_y[train_X.mean(axis = 1) > 0.5 - task_params['p']]
    else:
        train_X = None
        train_X_biased = None
        train_y_biased = 0
        train_y = 1

    valid_X_biased = create_test_matrix_11(int(task_params['valid_size'] * task_params['rho']), dimension, p=task_params['p'])
    valid_X_uniform = create_test_matrix_11(int(task_params['valid_size'] * (1 - task_params['rho'])), dimension)
    valid_X = np.vstack([valid_X_biased, valid_X_uniform])
    valid_y_biased = task_params['target_function'](valid_X_biased)
    valid_y = task_params['target_function'](valid_X)

   
    test_X_biased = create_test_matrix_11(int(task_params['test_size'] * task_params['rho']), dimension, p=task_params['p'])
    test_X_uniform = create_test_matrix_11(int(task_params['test_size'] * (1- task_params['rho'])), dimension)
    test_X = np.vstack([test_X_biased, test_X_uniform])
    test_y = task_params['target_function'](test_X)

    # Checking the samples
    # print(f"Shape of train samples: {train_X.shape}, valid samples: {valid_X.shape}, test samples: {test_X.shape}")

    # Running and saving the results
    if curriculum:
        print("Curriculum phase 1")
        iter_logs, train_losses, valid_losses, test_losses, train_accs, valid_accs, test_accs, coefficients, model = train(train_X_biased, train_y_biased, valid_X_biased, valid_y_biased, test_X, test_y, computation_interval=task_params['compute_int'], verbose_interval=task_params['verbose_int'], monomials=mask, print_coefficients=task_params['print_coefficients'], eps = 0.01)
        print("Curriculum phase 2")
        iter_logs_temp, train_losses_temp, valid_losses_temp, test_losses_temp, train_accs_temp, valid_accs_temp, test_accs_temp, coefficients_temp, _ = train(train_X, train_y, valid_X, valid_y, test_X, test_y, computation_interval=task_params['compute_int'], verbose_interval=task_params['verbose_int'], monomials=mask, print_coefficients=task_params['print_coefficients'], model=model, eps = task_params['eps'])
        iter_logs += iter_logs_temp
        train_losses += train_losses_temp
        valid_losses += valid_losses_temp
        test_losses += test_losses_temp
        train_accs += train_accs_temp
        valid_accs += valid_accs_temp
        test_accs += test_accs_temp
        coefficients += coefficients_temp
    else: 
        print("Training without curriculum")
        iter_logs, train_losses, valid_losses, test_losses, train_accs, valid_accs, test_accs, coefficients, _ = train(train_X, train_y, valid_X, valid_y, test_X, test_y, computation_interval=task_params['compute_int'], verbose_interval=task_params['verbose_int'], monomials=mask, print_coefficients=task_params['print_coefficients'], eps = task_params['eps'])
    saved_data = {'iters': np.array(iter_logs), 'train_losses': train_losses, 
                  'valid_losses': valid_losses, 'test_losses': test_losses, 'train_accs': train_accs, 'valid_accs': valid_accs, 'test_accs': test_accs, 'coefficients': coefficients, 
                  'run_params': vars(args), 'curriculum': curriculum}
    
    with open(f"{args.task}_{task_params['model']}_{'' if task_params['loss'] == '' else task_params['loss'] + '_'}{task_params['seed']}_{task_params['lr']}_{task_params['opt']}_{curriculum}_{task_params['train_size']}_{task_params['p']}_{task_params['rho']}.npz", "wb") as f:
        np.savez(f, **saved_data)