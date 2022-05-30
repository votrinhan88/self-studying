'''
By information theory, 
    Entropy H: number of bits required to transmit a randomly selected event from a probability distribution.
        - If the probability distribution T is skewed (more predictable) => H is low
        - If the probability distribution T is smooth (more random)      => H is high
        - Formula: H(T) = (sum x for X states){-x.ln(t(x))}
            t(x): probability of state x in T

    Cross-Entropy (CE): Average number of total bits to represent an event from P (prediction) instead of T (target).
        - Syntax in PyTorch: F.cross_entropy(P, T)
        - If P is very different from T => Cross-entropy is high
        - Formula: CE(T|P) = F.cross_entropy(P, T) = (sum x for X states){-t(x).ln(p(x))}
            t(x): probability of state x in T
            p(x): probability of state x in P

    KL Divergence (KLD): Average number of extra bits to represent an event from P instead of T.
        - Syntax in PyTorch: F.kl_div(P, T)
        - If P is very different from T => KL Divergence is high
        - Formula: KLD(T|P) = F.kl_div(P, T) = (sum x for X states){-t(x).ln(p(x)/t(x))}
                                             = (sum x for X states){-t(x).[ln(p(x)) - ln(t(x))]}
                                             = CE(T|P) - H(P)
            => "Relative entropy"
            t(x): probability of state x in T
            p(x): probability of state x in P

Note that due to PyTorch's means of reduction, to give cross-entropy loss and KL divergence loss
in comparable amounts, one must use F.kl_div(..., reduction = 'batchmean').

This is due to the PyTorch's KL Div 'mean' reduction takes the average of the whole
NUM_SAMPLES*NUM_CLASSES entries, instead of the correct way: 'batchmean' takes the row average by 
sample, of sum of NUM_CLASSES in each column entries {whic is mean(..., dim = 1)}.

This issue has been addressed by PyTorch and will be fixed in a later update:

WARNING
reduction= “mean” doesn’t return the true KL divergence value, please use reduction= “batchmean”
which aligns with the mathematical definition. In a future release, “mean” will be changed to be
the same as “batchmean”.
'''

import torch
import torch.nn.functional as F

NUM_SAMPLES = 3
NUM_CLASSES = 10

def cross_entropy(input, target = None):
    # No target --> Cross-entropy of input with itself --> Entropy
    if target is None:
        target = F.softmax(input, dim = 1)

    # Input is converted to probability from logs
    # Target by default is in propability
    probability = F.softmax(input, dim = 1)

    ce = torch.zeros(NUM_SAMPLES)
    for sample_id in range(NUM_SAMPLES):
        ce[sample_id] = 0
        for class_id in range(NUM_CLASSES):
            ce[sample_id] += -target[sample_id, class_id] * torch.log(probability[sample_id, class_id])
    return ce

# Hard targets --> Probability distribution = one-hot vector
# input is in the logits domain
input = torch.randn([NUM_SAMPLES, NUM_CLASSES])
target_hard = torch.randint(size = [NUM_SAMPLES], low = 0, high = NUM_CLASSES)
target_onehot = F.one_hot(target_hard, num_classes = NUM_CLASSES)

# Verify that custom cross_entropy is the same to PyTorch's implementation (for hard targets):
CE_torch = F.cross_entropy(input, target_hard)
CE_cust = cross_entropy(input, target_onehot)
entropy_target_onehot = cross_entropy(torch.log(target_onehot))
KL_div = F.kl_div(F.log_softmax(input, dim = 1), target_onehot, reduction = 'batchmean')
KL_div_mean = F.kl_div(F.log_softmax(input, dim = 1), target_onehot, reduction = 'mean')
print('Hard targets:\n' +
      f"Self-implemented cross-entropy loss: {CE_cust.mean()}\n" +
      f"PyTorch's cross-entropy loss: {CE_torch}\n" +
      f"One-hot target's entropy: 0 - Does not contain information\n" +
      f"PyTorch's KL Divergence loss: {KL_div}\n" +
      f"PyTorch's KL Divergence loss, reduction = 'mean': {KL_div_mean}\n" +
      f"\tMultiplying by NUM_CLASSES = {NUM_CLASSES} also gives the correct value: {KL_div_mean*NUM_CLASSES}\n")

# Soft targets --> Probability distribution ~= one-hot
target_soft = torch.softmax(torch.randn([NUM_SAMPLES, NUM_CLASSES]), dim = 1)

# In this case, PyTorch's cross_entropy() is not applicable
CE_cust = cross_entropy(input, target_soft)
entropy_target_soft = cross_entropy(torch.log(target_soft))
KL_div = F.kl_div(F.log_softmax(input, dim = 1), target_soft, reduction = 'batchmean')
KL_div_mean = F.kl_div(F.log_softmax(input, dim = 1), target_soft, reduction = 'mean')
print('Soft targets:\n' +
      f"Self-implemented cross-entropy loss: {CE_cust.mean()}\n" +
      f"Soft target's entropy: {entropy_target_soft.mean()}\n" +
      f"PyTorch's KL Divergence loss: {KL_div}\n" + 
      f"PyTorch's KL Divergence loss, reduction = 'mean': {KL_div_mean}\n" +
      f"\tMultiplying by NUM_CLASSES = {NUM_CLASSES} also gives the correct value: {KL_div_mean*NUM_CLASSES}")


pass    # For debugging purpose