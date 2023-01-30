import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn import datasets


def data_gen(num_samples = 20, num_features=1, std_dev=1, noise_dev=1, dist='gauss'):
  """
    data generating function for regression, supports Gauss and Poisson distribution
  Args:
    num_samples: number of samples to generate
    num_features: number of features/ dimensions of our datapoints, will be interesting in extension part
    std_dev: standard deviation for Gauss sampling
    noise_dev: standard deviation for Gaussian noise (delta)
    dist: the desired distribution: 'gauss' vs. 'poisson'
  Returns:
    num_samples amount of datapoints with respective labels, distributed by given dist
  """
  if dist == 'gauss':
    x = np.random.normal(loc=0, scale=std_dev, size=num_samples)
    delta = np.random.normal(loc=0, scale=noise_dev, size=num_samples)
    #w = 1, so it suffices to add x and delta
    y = x + delta
    return torch.Tensor(x).view(-1,1), torch.Tensor(y).view(-1,1)

  if dist == 'poisson':
    x = np.random.poisson(5.0, size=num_samples) + np.ones(num_samples)
    delta = np.random.normal(loc=0, scale=noise_dev, size=num_samples)
    y = x + delta
    return torch.Tensor(x).view(-1,1), torch.Tensor(y).view(-1,1)


def train_lin_reg_model(eps_list = [0.1, 0.2, 0.5, 1.0], num_repetitions=100, dist='gauss', n_points = list(range(1, 20,2)), points_step_size = 2):
    """
    Train the linear regression model for different values of epsilon. eps=0 means no adversarial perturbation. 
    The model consists of just one linear layer without any non-linearity or bias.
  Args:
    eps_list: eps values for which the model is to be trained
    num_repetition: the number of repetitions we want of our training process before averaging, 
      more repetitions -> smoother results
    dist: the distribution to sample data points from
    n_points: the different train dataset sizes we want to investigate for the training process
    points_step_size: set this the value between two consecutive vals in n_points
  Returns:
    A numpy array containing the test loss values for each eps value, each trainset size and each repetition 
  """
    n_test_points = 5000
    adv_test_loss = np.zeros([len(eps_list) + 1, len(n_points), num_repetitions], dtype=np.float64)
    epochs = 100
    model = torch.nn.Linear(1,1, bias= False)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss = torch.nn.MSELoss()

    #benign model
    for n in n_points:
        for r in range(1,num_repetitions):
            model.reset_parameters()

            #generate train data
            x, y = data_gen(num_samples=n, dist=dist)

            #standard model with optimal weights
            optimal_model_weight = 1/(x.T @ x) * (x.T @ y)

            #generate test data
            x_test, y_test = data_gen(num_samples=n_test_points, dist= dist)
            test_loss_benign = loss(y_test, torch.inner(optimal_model_weight, x_test))

            n_ind = (n-1) // points_step_size
            adv_test_loss [0, n_ind, 0]=int(n)
            adv_test_loss [0, n_ind, r]=test_loss_benign

    #adv model for different vals of eps
    for eps_idx in range(1, len(eps_list) + 1):
        eps = eps_list[eps_idx-1]

        for n in n_points:
            for r in range(1,num_repetitions):
                model.reset_parameters()
                x, y = data_gen(num_samples=n, dist=dist)
                # minimize term from Observation 5 in Chen et al. 'more data can expand..' [2020]
                for epoch in range(0, epochs):
                    w = 0.1
                    # we need this to be able to apply the l1 norm to model weights
                    for m in model.parameters():
                        weights_rob = torch.mean(torch.pow(torch.abs(y-model(x.float())) + eps * torch.norm(m, 1),2))
                        opt.zero_grad()
                        weights_rob.backward()
                        opt.step()
                        break


                #calculate the loss
                x_test, y_test = data_gen(num_samples=n_test_points, dist=dist)
                test_loss_adv = loss(y_test, torch.inner(weights_rob, x_test))


                n_ind = (n-1) // points_step_size
                adv_test_loss [eps_idx, n_ind, 0]=int(n)
                adv_test_loss [eps_idx, n_ind, r]=test_loss_adv
                

def plot_reg_loss(eps_list, adv_test_loss, training_points = list(range(1, 21))):
    x = training_points
    for eps_idx in range(len(eps_list) + 1):
        if eps_idx > 0:
            eps = eps_list[eps_idx-1]
        else:
            eps = 0
        test_eps=np.zeros(len(x))
    
        for n in x:
            n=int(n)
            n_ind = (n-1)//2
            test_eps[n_ind]=np.mean(adv_test_loss[eps_idx,n_ind,1:])
            
        plt.plot(x, test_eps,label="$\epsilon$ = {}".format(eps))   
    plt.legend(loc='upper right')
    plt.xlabel('Size of training dataset')
    plt.ylabel('Test loss')
    plt.yscale('log')
    plt.show()
