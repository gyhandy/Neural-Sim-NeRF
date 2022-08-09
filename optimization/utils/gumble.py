from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_probs(cats, probs):  # real probability
    plt.bar(cats, probs)
    plt.xlabel("Category")
    plt.ylabel("Original Probability")



def plot_estimated_probs(samples, ylabel=''):
    n_cats = np.max(samples) + 1
    estd_probs, _, _ = plt.hist(samples, bins=np.arange(n_cats + 1), align='left', edgecolor='white')
    plt.xlabel('Category')
    plt.ylabel(ylabel + 'Estimated probability')
    return estd_probs


def print_probs(probs):
    print(probs)




######################################

def sample_gumbel(logits): # use argmax not differentiable
    noise = np.random.gumbel(size=len(logits))
    sample = np.argmax(logits + noise)
    return sample




def sample_uniform(differentiable_samples):
    uniform_output = []
    for i, sample in enumerate(differentiable_samples):
        uniform_output.append(np.random.uniform(sample - 22.5, sample + 22.5))
    return uniform_output


#######################################

def softmax(logits):
    return np.exp(logits) / np.sum(np.exp(logits))


# def differentiable_sample(logits, degrees, temperature=.5):
#     noise = torch.Tensor(np.random.gumbel(size=len(logits)))
#     m = torch.nn.Softmax(dim=0)
#     logits_with_noise = m((logits + noise) / temperature)
#     # print(logits_with_noise)
#     sample = torch.sum(logits_with_noise * degrees) # times category degree means select
#     return sample
def differentiable_sample(logits, degrees, gumbel_noise, temperature=.5): # provide gumble noise
    noise = torch.Tensor(gumbel_noise)
    m = torch.nn.Softmax(dim=0)
    logits_with_noise = m((logits + noise) / temperature)
    # print(logits_with_noise)
    sample = torch.sum(logits_with_noise * degrees) # times category degree means select
    return sample
def differentiable_sample_nograd(logits, degrees, temperature=.5):
    # noise = np.random.gumbel(size=len(logits)).astype('float16') # keep only the last 4 digit
    noise = np.random.gumbel(size=len(logits))
    logits_with_noise = softmax((logits + noise) / temperature)
    # print(logits_with_noise)
    sample = np.sum(logits_with_noise * degrees) # times category degree means select
    return sample, noise
# def round_category(samples):
#     # degrees = np.array([0, 45, 90, 135, 180, 225, 270, 315]) devide 45 and then
#     for sample in samples:
#         category = sample / 45 + 1

def plot_estimated_probs_(samples, ylabel=''):
    samples = np.rint(samples)
    n_cats = np.max(samples) + 1
    estd_probs, _, _ = plt.hist(samples, bins=np.arange(n_cats + 1), align='left', edgecolor='white')
    plt.xlabel('Category')
    plt.ylabel(ylabel + 'Estimated probability')
    return estd_probs


if __name__=='__main__':
    # real distribution setting
    n_cats = 8
    n_samples = 1000
    cats = np.arange(n_cats) # 1 represent 0 degree
    degrees = np.array([0, 45, 90, 135, 180, 225, 270, 315]) + 22.5
    # cats = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    # probs = np.random.randint(low=1, high=20, size=n_cats)
    # probs = probs / sum(probs)
    # probs = np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
    probs = np.array([0, 0.3, 0, 0, 0, 0, 0.7, 0])
    logits = np.log(probs)

    # sample with np.random
    samples = np.random.choice(cats, p=probs, size=n_samples)  # sample based on probability
    # sample with Gaumble max
    gumbel_samples = [sample_gumbel(logits) for _ in range(n_samples)]
    # sample with  Gumble softmax
    differentiable_samples_1 = [differentiable_sample(logits, degrees, 0.01) for _ in range(n_samples)]
    # differentiable_samples_1_uniform = sample_uniform(differentiable_samples_1)
    # differentiable_samples_1_uniform = [np.random.uniform(sample-22.5, sample+22.5) for sample in differentiable_samples_1]
    differentiable_samples_1_uniform = [sample-22.5 + 45*np.random.uniform(0, 1) for sample in differentiable_samples_1] # uniform reparameterization
    differentiable_samples_1 = [int(sample / 45) for sample in differentiable_samples_1_uniform] # assign range based on the degree for vis
    differentiable_samples_2 = [differentiable_sample(logits, degrees, 0.1) for _ in range(n_samples)]
    differentiable_samples_2_uniform = [sample-22.5 + 45*np.random.uniform(0, 1) for sample in differentiable_samples_2]
    differentiable_samples_2 = [int(sample / 45) for sample in differentiable_samples_2_uniform]  # assign range based on the degree
    differentiable_samples_3 = [differentiable_sample(logits, degrees, 5) for _ in range(n_samples)]
    differentiable_samples_3_uniform = [sample-22.5 + 45*np.random.uniform(0, 1) for sample in differentiable_samples_3]
    differentiable_samples_3 = [int(sample / 45) for sample in differentiable_samples_3_uniform]  # assign range based on the degree

    plt.figure(figsize=(24, 4))
    # plt.figure()
    plt.subplot(1, 6, 1)
    plot_probs() # original probability
    plt.subplot(1, 6, 2)
    estd_probs = plot_estimated_probs(samples) #np.random.choice(cats, p=probs, size=n_samples)  # sample based on probability
    plt.subplot(1, 6, 3)
    gumbel_estd_probs = plot_estimated_probs(gumbel_samples, 'Gumbel ') # np.argmax(logits + noise)
    plt.subplot(1, 6, 4)
    gumbelsoft_estd_probs_1 = plot_estimated_probs_(differentiable_samples_1, 'Gumbel softmax t=0.01')
    plt.subplot(1, 6, 5)
    gumbelsoft_estd_probs_2 = plot_estimated_probs_(differentiable_samples_2, 'Gumbel softmax t=0.1')
    plt.subplot(1, 6, 6)
    gumbelsoft_estd_probs_3 = plot_estimated_probs_(differentiable_samples_3, 'Gumbel softmax t=5')
    plt.tight_layout()
    plt.savefig('gumbel')

    print('Gumbel Softmax Estimated probabilities:\t', end='')
    # print_probs(gumbelsoft_estd_probs_1)
    plt.show()
