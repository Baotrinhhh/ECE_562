from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Flatten input: x in R^{N x d_1 x ... x d_k} -> x_flat in R^{N x D} where D = d_1 x ... x d_k
    x_flat = x.reshape(x.shape[0], -1)

    # Affine transformation: out = x_flat @ w + b in R^{N x M}
    out = x_flat @ w + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M) 
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = x.shape[0]
    # Flatten input for gradient computation: x in R^{N x d_1 x ... x d_k} -> x_flat in R^{N x D}
    x_flat = x.reshape(N, -1)

    # Chain rule for affine layer gradients:
    # Given: out = x_flat @ w + b, and upstream gradient dout = dL/dout
    
    # dL/dx = dL/dout @ dout/dx = dout @ w^T, then reshape to original x shape
    dx = dout.dot(w.T).reshape(x.shape)
    
    # dL/dw = dL/dout @ dout/dw = x_flat^T @ dout
    dw = x_flat.T.dot(dout)
    
    # dL/db = dL/dout @ dout/db = sum(dout) over batch dimension
    db = dout.sum(axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # ReLU activation: f(x) = max(0, x)
    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # ReLU derivative: f'(x) = 1 if x > 0, else 0
    # Apply chain rule: dL/dx = dL/dout * dout/dx = dout * (x > 0)
    dx = dout * (x > 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Compute sample mean and variance over the minibatch
        # mu_B = (1/m) * sum_{i=1}^m x_i   where m = N (batch size)
        sample_mean = np.mean(x, axis=0)  # mu_B in R^D, Shape: (D,)

        # sigma^2_B = (1/m) * sum_{i=1}^m (x_i - mu_B)^2
        sample_var = np.var(x, axis=0)    # sigma^2_B in R^D, Shape: (D,)

        # Center the data
        # x_hat_i = x_i - mu_B
        x_centered = x - sample_mean      # x_hat in R^{NxD}, Shape: (N, D)
        
        # Normalize by standard deviation
        # x_tilde_i = x_hat_i / sqrt(sigma^2_B + eps) = (x_i - mu_B) / sqrt(sigma^2_B + eps)
        x_norm = x_centered / np.sqrt(sample_var + eps)  # x_tilde in R^{NxD}, Shape: (N, D)
        
        # Scale and shift (learnable affine transformation)
        # y_i = gamma * x_tilde_i + beta   
        out = gamma * x_norm + beta       # y in R^{NxD}, Shape: (N, D)
        
        #  Update running statistics using exponential moving average
        # mu_running <- alpha*mu_running + (1-alpha)*mu_B   where alpha = momentum
        # sigma^2_running <- alpha*sigma^2_running + (1-alpha)*sigma^2_B
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        # Cache intermediate values for backward pass computation
        # Need: x, x_hat, x_tilde, mu_B, sigma²_B, gamma, beta, eps for gradient calculations
        cache = (x, x_centered, x_norm, sample_mean, sample_var, gamma, beta, eps)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Test-time batch normalization uses pre-computed running statistics
        # Normalize using running statistics
        # x_tilde_i = (x_i - mu_running) / sqrt(sigma²_running + eps)
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        
        # Apply learnable affine transformation
        # y_i = gamma * x_tilde_i + beta
        out = gamma * x_norm + beta
        
        # No cache needed for test mode 
        cache = None

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Extract cached values from forward pass
    x, x_centered, x_norm, sample_mean, sample_var, gamma, beta, eps = cache
    N, D = x.shape
    
    # Gradient w.r.t. beta (bias parameter)
    # dL/dbeta = dL/dy * dy/dbeta = dL/dy * 1 = sum(dL/dy_i) over all i
    # Since y_i = gamma*x_tilde_i + beta, we have dy_i/dbeta = 1
    dbeta = np.sum(dout, axis=0)  # Shape: (D,)
    
    # Gradient w.r.t. gamma (scale parameter)  
    # dL/dgamma = dL/dy * dy/dgamma = dL/dy * x_tilde = sum(dL/dy_i * x_tilde_i) over all i
    # Since y_i = gamma*x_tilde_i + beta, we have dy_i/dgamma = x_tilde_i
    dgamma = np.sum(dout * x_norm, axis=0)  # Shape: (D,)
    
    # Gradient w.r.t. x_tilde (normalized input)
    # dL/dx_tilde = dL/dy * dy/dx_tilde = dL/dy * gamma
    # Since y_i = gamma*x_tilde_i + beta, we have dy_i/dx_tilde_i = gamma
    dx_norm = dout * gamma  # Shape: (N, D)
    
    # Gradient w.r.t. sample variance sigma^2_B
    # x_tilde_i = x_hat_i / sqrt(sigma^2_B + eps), so dx_tilde_i/dsigma^2_B = -1/2 * x_hat_i * (sigma^2_B + eps)^(-3/2)
    # dL/dsigma^2_B = sum(dL/dx_tilde_i * dx_tilde_i/dsigma^2_B) = -1/2 * sum(dL/dx_tilde_i * x_hat_i) * (sigma^2_B + eps)^(-3/2)
    dvar = np.sum(dx_norm * x_centered, axis=0) * -0.5 * np.power(sample_var + eps, -1.5)  # Shape: (D,)
    
    # Gradient w.r.t. x_hat (centered input)
    # Two paths: direct through normalization and indirect through variance
    # Path 1: dL/dx_hat via normalization: dx_tilde_i/dx_hat_i = 1/sqrt(sigma^2_B + eps)
    # Path 2: dL/dx_hat via variance: dsigma^2_B/dx_hat_i = 2*x_hat_i/N (since sigma^2_B = (1/N)*sum(x_hat_i^2))
    dx_centered = dx_norm / np.sqrt(sample_var + eps) + dvar * 2.0 * x_centered / N  # Shape: (N, D)
    
    # Gradient w.r.t. sample mean mu_B  
    # x_hat_i = x_i - mu_B, so dx_hat_i/dmu_B = -1
    # dL/dmu_B = sum(dL/dx_hat_i * dx_hat_i/dmu_B) = -sum(dL/dx_hat_i)
    dmean = -np.sum(dx_centered, axis=0)  # Shape: (D,)
    
    # Gradient w.r.t. input x
    # Two paths: direct through centering and indirect through mean
    # Path 1: dL/dx via centering: dx_hat_i/dx_i = 1  
    # Path 2: dL/dx via mean: dmu_B/dx_i = 1/N (since mu_B = (1/N)*sum(x_i))
    dx = dx_centered + dmean / N  # Shape: (N, D)
    
    # Mathematical summary of the computation graph:
    # x -> mu_B, sigma²_B -> x_hat -> x_tilde -> y
    # Gradients flow backward: dL/dy -> dL/dx_tilde -> dL/dx_hat -> dL/dx
    #                                     |         |
    #                                 dL/dsigma²_B -> dL/dmu_B

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Extract cached values from forward pass
    x, x_centered, x_norm, sample_mean, sample_var, gamma, beta, eps = cache
    N, D = x.shape
    
    # Same gradients for beta and gamma as in standard implementation
    # dL/dbeta = sum(dL/dy_i) - simple sum over all upstream gradients
    dbeta = np.sum(dout, axis=0)  # Shape: (D,)
    
    # dL/dgamma = sum(dL/dy_i * x_tilde_i) - weighted sum with normalized inputs
    dgamma = np.sum(dout * x_norm, axis=0)  # Shape: (D,)
    
    # Gradient w.r.t. normalized inputs
    # dL/dx_tilde = dL/dy * gamma
    dx_norm = dout * gamma  # Shape: (N, D)
    
    # Simplified gradient w.r.t. centered inputs
    # Mathematical derivation of the alternative formula:
    # Let sigma = sqrt(sigma²_B + eps), then x_tilde_i = x_hat_i / sigma
    # 
    # Working backward from x_tilde to x_hat:
    # dL/dx_hat_i has two components:
    # 1. Direct: dx_tilde_i/dx_hat_i = 1/sigma
    # 2. Through variance: dsigma/dx_hat_i affects all x_tilde_j via sigma
    #
    # The key insight: we can factor out common terms and simplify
    # After algebraic manipulation, the gradient becomes:
    # dL/dx_hat = (1/sigma) * [dL/dx_tilde - (1/N)*sum(dL/dx_tilde_j) - (1/N)*sum(dL/dx_tilde_j * x_tilde_j)*x_tilde]
    
    # Alternative simplified formula:
    dx_centered = (dx_norm - dx_norm.mean(axis=0) - x_norm * (dx_norm * x_norm).mean(axis=0)) / np.sqrt(sample_var + eps)
    
    # Final gradient w.r.t. input x
    # From x_hat_i = x_i - mu_B, we get:
    # dL/dx = dL/dx_hat - (1/N) * sum(dL/dx_hat_j)  [chain rule through mean]
    dx = dx_centered - dx_centered.mean(axis=0)
    
    # Mathematical explanation of the simplified formula:
    # The key insight is that all the gradient paths through variance and mean
    # can be combined into the compact expression above. This avoids computing
    # intermediate gradients dvar and dmean separately, leading to a more
    # efficient and numerically stable implementation.

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Key insight: Layer norm normalizes across features (axis=1) instead of samples (axis=0)
    # We can transpose the input, apply batch norm logic, then transpose back!
    # 
    # Batch norm: normalize across samples for each feature
    # Layer norm: normalize across features for each sample
    # 
    # Mathematical transformation:
    # If we transpose x from (N,D) → (D,N), then:
    # - Batch norm on transposed data normalizes across the N dimension 
    # - This is equivalent to layer norm on original data across D dimension
    
    N, D = x.shape
    
    # Transpose to convert layer norm → batch norm problem
    # x.T shape: (D, N) - now each "sample" is a feature, each "feature" is a data point
    x_T = x.T  # Shape: (D, N)
    
    # Apply batch normalization logic on transposed data
    # Compute mean and variance across the "batch" dimension (axis=0)
    # This computes statistics across all data points for each feature
    sample_mean = np.mean(x_T, axis=0)  # mu in R^N, mean for each data point
    sample_var = np.var(x_T, axis=0)    # sigma² in R^N, variance for each data point
    
    # Normalize the transposed data
    # x_hat = (x - mu) / sqrt(sigma² + eps)
    x_T_centered = x_T - sample_mean     # Shape: (D, N)
    x_T_norm = x_T_centered / np.sqrt(sample_var + eps)  # Shape: (D, N)
    
    # Transpose back to original orientation
    x_centered = x_T_centered.T          # Shape: (N, D)
    x_norm = x_T_norm.T                  # Shape: (N, D)
    
    # Apply learnable affine transformation
    # y = gamma * x_tilde + beta  (same as batch norm)
    out = gamma * x_norm + beta          # Shape: (N, D)
    
    # Cache values needed for backward pass
    # Note: we store the non-transposed versions for easier backward computation
    cache = (x, x_centered, x_norm, sample_mean, sample_var, gamma, beta, eps)
    
    # Mathematical summary:
    # Layer norm per sample: mu_i = (1/D)*sum_j x_ij, sigma²_i = (1/D)*sum_j(x_ij - mu_i)²
    # Normalization: x_tilde_ij = (x_ij - mu_i) / sqrt(sigma²_i + eps)
    # Final output: y_ij = gamma_j * x_tilde_ij + beta_j

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Extract cached values from forward pass
    x, x_centered, x_norm, sample_mean, sample_var, gamma, beta, eps = cache
    N, D = x.shape
    
    # Key insight: Use the same transpose trick as in forward pass
    # Layer norm normalizes across features (D) for each sample (N)
    # We can transpose and use batch norm backward logic, then transpose back
    
    # Gradient w.r.t. beta (shift parameter)
    # dL/dbeta = dL/dy * dy/dbeta = dL/dy * 1 = sum(dL/dy_i) over all samples
    # Since y_ij = gamma_j * x_tilde_ij + beta_j, we have dy_ij/dbeta_j = 1
    dbeta = np.sum(dout, axis=0)  # Shape: (D,)
    
    # Gradient w.r.t. gamma (scale parameter)
    # dL/dgamma = dL/dy * dy/dgamma = dL/dy * x_tilde = sum(dL/dy_ij * x_tilde_ij) over all samples
    # Since y_ij = gamma_j * x_tilde_ij + beta_j, we have dy_ij/dgamma_j = x_tilde_ij
    dgamma = np.sum(dout * x_norm, axis=0)  # Shape: (D,)
    
    # For dx, we need to apply the transpose trick
    # Transpose dout and apply batch norm backward logic, then transpose back
    dout_T = dout.T  # Shape: (D, N)
    x_centered_T = x_centered.T  # Shape: (D, N)
    
    # Apply batch norm backward logic on transposed data
    # Gradient w.r.t. normalized input (transposed)
    dx_norm_T = dout_T * gamma[:, np.newaxis]  # Shape: (D, N), broadcast gamma
    
    # Gradient w.r.t. variance (for each sample, computed across features)
    # In layer norm: sigma^2_i = (1/D) * sum_j (x_ij - mu_i)^2
    # dsigma^2_i/dx_hat_ij = 2*x_hat_ij/D
    dvar_T = np.sum(dx_norm_T * x_centered_T, axis=0) * -0.5 * np.power(sample_var + eps, -1.5)  # Shape: (N,)
    
    # Gradient w.r.t. centered input (transposed)
    # Path 1: direct through normalization
    # Path 2: through variance
    dx_centered_T = dx_norm_T / np.sqrt(sample_var + eps) + dvar_T * 2.0 * x_centered_T / D  # Shape: (D, N)
    
    # Gradient w.r.t. mean (for each sample, computed across features)
    # In layer norm: mu_i = (1/D) * sum_j x_ij
    # dmu_i/dx_ij = 1/D
    dmean_T = -np.sum(dx_centered_T, axis=0)  # Shape: (N,)
    
    # Final gradient w.r.t. input (transposed)
    # Path 1: direct through centering
    # Path 2: through mean  
    dx_T = dx_centered_T + dmean_T / D  # Shape: (D, N)
    
    # Transpose back to original orientation
    dx = dx_T.T  # Shape: (N, D)
    
    # Mathematical summary for layer normalization:
    # For each sample i: mu_i = (1/D)*sum_j x_ij, sigma^2_i = (1/D)*sum_j(x_ij - mu_i)²
    # Normalization: x_tilde_ij = (x_ij - mu_i) / sqrt(sigma^2_i + eps)
    # Output: y_ij = gamma_j * x_tilde_ij + beta_j
    # The transpose trick allows us to reuse batch norm backward computation

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Inverted dropout implementation:
        # 1. Create a random mask with probability p of keeping each neuron
        # 2. Scale the kept neurons by 1/p to maintain expected value
        # This ensures test-time behavior doesn't need scaling
        
        # Generate random mask: 1 where we keep neurons (prob p), 0 where we drop
        mask = (np.random.rand(*x.shape) < p).astype(x.dtype)
        
        # Apply inverted dropout: multiply by mask and scale by 1/p
        # This maintains the expected value of activations
        out = x * mask / p

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Test phase: no dropout, just pass input through unchanged
        # The inverted scaling during training ensures we don't need 
        # any scaling at test time
        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Apply the same mask and scaling as forward pass
        # Chain rule: dL/dx = dL/dout * dout/dx = dout * mask / p
        dx = dout * mask / dropout_param["p"]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Extract convolution parameters
    pad = conv_param['pad']
    stride = conv_param['stride']
    
    # Input and filter dimensions
    N, C, H_in, W_in = x.shape    # input: (batch_size, channels, height, width)
    F, _, H_filter, W_filter = w.shape  # filters: (num_filters, channels, height, width)
    
    # Calculate output dimensions using convolution formula
    H_out = 1 + (H_in + 2 * pad - H_filter) // stride
    W_out = 1 + (W_in + 2 * pad - W_filter) // stride

    # Helper function to create sliding window views for efficient convolution
    create_windows = lambda x: np.lib.stride_tricks.sliding_window_view(x, (W_filter, H_filter, C, N))

    # Reshape filters to matrix form for efficient computation: (F, C*H_filter*W_filter)
    w_matrix = w.reshape(F, -1)
    
    # Apply zero padding to input: (N, C, H_in+2*pad, W_in+2*pad)
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    
    # Create column matrix from input windows: (N, C*H_filter*W_filter, H_out*W_out)
    x_windows = create_windows(x_padded.T).T[..., ::stride, ::stride].reshape(N, C*H_filter*W_filter, -1)

    # Perform convolution: w_matrix @ x_windows + bias
    # Result shape: (N, F, H_out, W_out)
    out = (w_matrix @ x_windows).reshape(N, F, H_out, W_out) + np.expand_dims(b, axis=(2, 1))
    
    # Store padded input for backward pass
    x = x_padded

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Helper function for creating sliding window views
    create_windows = np.lib.stride_tricks.sliding_window_view

    # Extract cached parameters
    x_padded, w, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']
    F, C, H_filter, W_filter = w.shape
    N, _, H_out, W_out = dout.shape
    
    # Upsample dout by inserting zeros for strided convolution
    # Insert (stride-1) zeros between each element to "undo" the stride
    dout_upsampled = np.insert(dout, [*range(1, H_out)] * (stride-1), 0, axis=2)  # rows
    dout_upsampled = np.insert(dout_upsampled, [*range(1, W_out)] * (stride-1), 0, axis=3)  # columns
    
    # Pad dout for full convolution (correlation actually)
    dout_padded = np.pad(dout_upsampled, ((0,), (0,), (H_filter-1,), (W_filter-1,)), 'constant')

    # Create sliding window views for gradient computation
    x_windows = create_windows(x_padded, (N, C, dout_upsampled.shape[2], dout_upsampled.shape[3]))
    dout_windows = create_windows(dout_padded, (N, F, H_filter, W_filter))
    
    # Rotate weights 180 degrees for convolution (equivalent to correlation with flipped kernel)
    w_rotated = np.rot90(w, 2, axes=(2, 3))

    # Compute gradients using Einstein summation notation
    db = np.einsum('ijkl->j', dout)  # Sum over all dimensions except filter dimension
    dw = np.einsum('ijkl,mnopiqkl->jqop', dout_upsampled, x_windows)  # Cross-correlation
    dx = np.einsum('ijkl,mnopqikl->qjop', w_rotated, dout_windows)[..., pad:-pad, pad:-pad]  # Convolution

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Extract pooling parameters
    stride = pool_param['stride']
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    
    # Input dimensions
    N, C, H_in, W_in = x.shape
    
    # Calculate output dimensions using pooling formula
    H_out = 1 + (H_in - pool_height) // stride
    W_out = 1 + (W_in - pool_width) // stride

    # Helper function to create sliding window views (requires numpy >= 1.20)
    create_windows = lambda x: np.lib.stride_tricks.sliding_window_view(x, (pool_width, pool_height, C, N))

    # Create sliding windows and apply max pooling
    # Shape after windowing: (N, C, pool_height*pool_width, H_out*W_out)
    x_windows = create_windows(x.T).T[..., ::stride, ::stride].reshape(N, C, pool_height*pool_width, -1)
    
    # Take maximum over each pooling window and reshape to output dimensions
    out = x_windows.max(axis=2).reshape(N, C, H_out, W_out)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Extract cached parameters
    x, pool_param = cache
    N, C, H_out, W_out = dout.shape
    dx = np.zeros_like(x)  # Initialize gradient tensor

    # Extract pooling parameters  
    stride = pool_param['stride']
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']

    # Iterate through each position in the output feature map
    for i in range(H_out):
        for j in range(W_out):
            # Calculate input window coordinates
            h_start, w_start = i * stride, j * stride
            h_end, w_end = h_start + pool_height, w_start + pool_width
            
            # Create index arrays for batch and channel dimensions
            batch_indices, channel_indices = np.indices((N, C))
            
            # Extract the pooling window from input
            pool_window = x[:, :, h_start:h_end, w_start:w_end].reshape(N, C, -1)
            
            # Find the index of maximum value in each pooling window
            max_indices = np.argmax(pool_window, axis=2)
            max_h, max_w = np.unravel_index(max_indices, (pool_height, pool_width))
            
            # Propagate gradient only to the maximum element in each window
            dx[batch_indices, channel_indices, h_start + max_h, w_start + max_w] += dout[batch_indices, channel_indices, i, j]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    # Reshape to (N*H*W, C) to treat each spatial location as a separate sample
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)
    # Apply vanilla batch normalization
    out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    # Reshape back to (N, C, H, W)
    out = out_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    # Reshape dout to (N*H*W, C)
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    # Apply vanilla batch normalization backward pass
    dx_reshaped, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
    # Reshape dx back to (N, C, H, W)
    dx = dx_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    # Reshape to (N*G, C//G, H, W) to group channels, then treat as layer norm problem
    x_grouped = x.reshape(N, G, C//G, H, W)
    # Reshape to (N*G, C//G*H*W) for layer normalization within each group
    x_reshaped = x_grouped.reshape(N*G, -1)
    
    # Create dummy gamma and beta for layer norm (we'll apply real gamma/beta later)
    dummy_gamma = np.ones(C//G * H * W)
    dummy_beta = np.zeros(C//G * H * W)
    ln_param = {'eps': eps}
    
    # Apply layer normalization within each group
    out_reshaped, cache_ln = layernorm_forward(x_reshaped, dummy_gamma, dummy_beta, ln_param)
    
    # Reshape back to (N, C, H, W)
    out_normalized = out_reshaped.reshape(N, G, C//G, H, W).reshape(N, C, H, W)
    
    # Apply per-channel gamma and beta scaling
    out = gamma.reshape(1, C, 1, 1) * out_normalized + beta.reshape(1, C, 1, 1)
    
    # Store cache with original shapes and parameters
    cache = (x, out_normalized, gamma, beta, G, cache_ln)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, out_normalized, gamma, beta, G, cache_ln = cache
    N, C, H, W = dout.shape
    
    # Gradient w.r.t. gamma and beta (same as batch norm)
    dgamma = np.sum(dout * out_normalized, axis=(0, 2, 3))  # Sum over N, H, W dimensions
    dbeta = np.sum(dout, axis=(0, 2, 3))  # Sum over N, H, W dimensions
    
    # Gradient w.r.t. normalized output
    dout_normalized = dout * gamma.reshape(1, C, 1, 1)
    
    # Reshape for layer norm backward pass
    dout_reshaped = dout_normalized.reshape(N, G, C//G, H, W).reshape(N*G, -1)
    
    # Apply layer norm backward pass
    dx_reshaped, _, _ = layernorm_backward(dout_reshaped, cache_ln)
    
    # Reshape back to original input shape
    dx = dx_reshaped.reshape(N, G, C//G, H, W).reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
