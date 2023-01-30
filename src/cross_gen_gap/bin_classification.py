import torch
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d


def data_gen(num_samples = 20, num_features=1, std_dev=2, dist='gauss', t=0):
  """
    data generating function for classification, supports Gauss and Bernoulli distribution.
  Args:
    num_samples: number of samples to generate
    num_features: number of features/ dimensions of our datapoints, will be interesting in extension part
    std_dev: standard deviation for Gauss sampling
    dist: the desired distribution: 'gauss' vs. 'bern'
    t: if 'bern' is picked as distribution, the val for tau can be given here
  Returns:
    num_samples amount of datapoints with respective labels, distributed by given dist
  """

  if dist == 'gauss':
    # out-of-the-box solution from sklearn to sample from Gaussian, np.full is used to support higher dim data sampling
    # for extension part
    x, y = datasets.make_blobs(n_samples=num_samples, n_features=num_features,
                           centers=[np.full((num_features),-1) ,np.full((num_features),1)],cluster_std= std_dev)
    y[y==0] = -1
    return torch.FloatTensor(x), torch.FloatTensor(y)
    
  elif dist == 'bern': 
    y = np.ones(num_samples)
    # we actually want exactly half of the samples from each class as this stabilizes the process, 
    # with enough repetitions, this would not make a difference
    y[:num_samples//2:] = -1
    x = torch.zeros([num_samples, num_features])
    for i in range(num_samples):
      ran = np.random.choice([y[i], -y[i]], p=[(1.0+t)/2.0, (1.0-t)/2.0])
      x[i] = ran
    
    #if we want to shuffle the arrays
    p = np.random.permutation(num_samples)
    x_perm = x[p]
    y_perm = y[p]
    return torch.FloatTensor(x_perm), torch.FloatTensor(y_perm)


class WeightClipper(object):
  """
  the weight clipper object is needed to model the boundary of |w| <= 1 in the classification setting;
  after each train epoch, we clamp w back to the desired region
  """

  # freq: how often do we want to clip weights, freq=1 == every step
  def __init__(self, frequency=1):
    self.frequency = frequency

  def __call__(self, module):
    # filter the variables to get the ones you want
    if hasattr(module, 'weight'):
      w = module.weight.data
      w = w.clamp(-1,1)
      module.weight.data = w


def epoch_robust(loader, model, epsilon, opt=None):
  """
    Iterate one training loop over dataset.
  Args:
    loader: DataLoader containing dataset
    model: model to adversarially train
    epsilon: adversarial strength
    opt: indicator if we want to optimize our model, opt=False -> eval mode
  Returns:
    training loss of current epoch

  adapted from https://adversarial-ml-tutorial.org/linear_models/
  """
  total_loss = 0.
  for X,y in loader:
    yp = model(X.view(X.shape[0], -1))[:,0] - y*epsilon*model.weight.norm(1)
    loss = torch.mean(torch.mul(-y, yp))
    if opt:
      opt.zero_grad()
      loss.backward()
      opt.step()
      
    total_loss += loss.item() * X.shape[0]
  return total_loss / len(loader.dataset)


def train_classification_model(eps_list, epoch_fn=epoch_robust, num_repetition=100, dist='gauss', n_points = list(range(1, 21)), points_step_size = 1):
  """
    Train the classification model for different values of epsilon. eps=0 means no adversarial perturbation. 
    The model consists of just one linear layer without any non-linearity or bias.
  Args:
    eps_list: eps values for which the model is to be trained
    epoch_fn: the function we want to use in order to train the model
    num_repetition: the number of repetitions we want of our training process before averaging, 
      more repetitions -> smoother results
    dist: the distribution to sample data points from
    n_points: the different train dataset sizes we want to investigate for the training process
    points_step_size: set this the value between two consecutive vals in n_points
  Returns:
    A numpy array containing the test loss values for each eps value, each trainset size and each repetition 
  """

  n_test_samples = 5000

  adv_test_loss = np.zeros([len(eps_list), len(n_points), num_repetition], dtype=np.float64)

  # we are using a linear model without bias
  model = torch.nn.Linear(1,1,bias=False)
  opt = torch.optim.SGD(model.parameters(), lr=1e-1)
  clipper = WeightClipper()

  for eps_idx in range(len(eps_list)):
    eps = eps_list[eps_idx]
    print('running training loop for epsilon = {}'.format(eps))
    for n in n_points: 
      
      for r in range(1,num_repetition,1):
        model.reset_parameters()
        x_train, y_train = data_gen(n, dist=dist)
        train_data = []
        for i in range(len(x_train)):
          train_data.append([x_train[i], y_train[i]])
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=x_train.size()[0])
        
        epochs = 100
        for i in range(epochs):
          train_loss = epoch_fn(train_loader, model, eps, opt)
          if i  % clipper.frequency == 0:
            model.apply(clipper)

        test_loss = 0
        x_test, y_test=data_gen(n_test_samples, dist=dist)
        test_data = []
        for i in range(len(x_test)):
          test_data.append([x_test[i], y_test[i]])
          test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=x_test.size()[0])
        
        test_loss = epoch_fn(test_loader, model,0)
        n_ind = (n-1)// points_step_size
        adv_test_loss [eps_idx, n_ind, 0]=int(n)
        adv_test_loss [eps_idx, n_ind, r]=test_loss
  return adv_test_loss


def plot_class_loss(eps_list, adv_test_loss, training_points = list(range(1, 21))):
  x= training_points

  for eps_idx in range(len(eps_list)):
    eps = eps_list[eps_idx]
    test_eps=np.zeros(len(x))
    
    for n in range(len(x)):
      n=int(n)
      test_eps[n-1]=np.mean(adv_test_loss[eps_idx,n-1,1:])

    ysmoothed = gaussian_filter1d(test_eps, sigma=2)
    plt.plot(x, ysmoothed,label="$\epsilon$ = {}".format(eps))   
  plt.legend(loc='upper right')
  plt.xlabel('Size of training dataset')
  plt.ylabel('Test loss')
  plt.show()

def plot_class_gap(eps_list, adv_test_loss, training_points = list(range(1, 21))):
    x = training_points
    for eps_idx in range(len(eps_list)):
        eps = eps_list[eps_idx]
        test_eps=np.zeros(len(x))
    
        for n in range(len(x)):
            n=int(n)
            test_eps[n-1]=np.mean(adv_test_loss[eps_idx,n-1,1:]) - np.mean(adv_test_loss[0,n-1,1:])
        
        ysmoothed = gaussian_filter1d(test_eps, sigma=2)
        plt.plot(x, ysmoothed,label="$\epsilon$ = {}".format(eps))
    plt.legend(loc='upper right')
    plt.xlabel('Size of training dataset')
    plt.ylabel('Cross generalization gap')
    plt.show()
