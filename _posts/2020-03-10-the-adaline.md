---
title: The ADALINE - Theory and Implementation of the First Neural Network Trained With Gradient Descent
published: true
mathjax: true
---

<iframe src="https://github.com/sponsors/pabloinsente/card" title="Sponsor pabloinsente" height="225" width="600" style="border: 0;"></iframe>

## Learning objectives

1. Understand the principles behind the creation of the ADALINE
2. Identify the similarities and differences between the perceptron and the ADALINE
3. Acquire an intuitive understanding of learning via gradient descent
4. Develop a basic code implementation of the ADALINE in Python
5. Determine what kind of problems can and can't be solved with the ADALINE 

## Historical and theoretical background

The ADALINE (*Adaptive Linear Neuron*) was introduced in 1959, shortly after Rosenblatt’s perceptron, by *Bernard Widrow* and *Ted Hoff* (one of the inventors of the microprocessor) at Stanford. Widrow and Hoff were electrical engineers, yet Widrow had attended the famous [Dartmouth workshop on artificial intelligence](https://en.wikipedia.org/wiki/Dartmouth_workshop) in 1956, an experience that got him interested in the idea of building brain-like artificial learning systems. When Widrow moved from MIT to Stanford, a colleague asked him whether he would be interested in taking Ted Hoff as his doctoral student. Widrow and Hoff came up with the ADALINE idea on a Friday during their first session working together. At the time, implementing an algorithm in a mainframe computer was slow and expensive, so they decided to build a small electronic device capable of being trained by the ADALINE algorithm to learn to classify patterns of inputs.  

The main difference between the perceptron and the ADALINE is that the later works by minimizing the [*mean squared error*](https://en.wikipedia.org/wiki/Mean_squared_error) of the predictions of a linear function. This means that the learning procedure is based on the outcome of a *linear function* rather than on the outcome of a threshold function as in the perceptron. **Figure 2** summarizes such difference schematically. 

From a cognitive science perspective, the main contribution of the ADALINE was methodological rather than theoretical. Widrow and Hoff were not primarily concerned with understanding the organization and function of the human mind. Although the ADALINE was initially applied to problems like speech and pattern recognition (Talbert et al., 1963), the main application of the ADALINE was in adaptive filtering and adaptive signal processing. Technologies like adaptive antennas, adaptive noise canceling, and adaptive equalization in high-speed modems (which makes Wifi works well), were developed by using the ADALINE (Widrow & Lehr, 1990).

Mathematically, learning from the output of a linear function enables the minimization of a continuous [*cost or loss function*](https://en.wikipedia.org/wiki/Loss_function). In simple terms, a cost function is a measure of the overall *badness* (or *goodness*) of the network predictions. Continuous cost functions have the advantage of having "nice" derivatives, that facilitate training neural nets by using the [chain rule of calculus](https://en.wikipedia.org/wiki/Chain_rule). This change opened the door to train more complex algorithms like non-linear multilayer perceptrons, logistic regression, support vector machines, and others. 

Next, we will review the ADALINE formalization, learning procedure, and optimization process.

## Mathematical formalization

Mathematically, the ADALINE is described by:

- a *linear function* that aggregates the input signal
- a *learning procedure* to adjust connection weights

Depending on the problem to be approached, a *threshold function*, as in the McCulloch-Pitts and the perceptron, can be added. Yet, such function is not part of the learning procedure, therefore, it is not strictly necessary to define an ADALINE.

### Linear aggregation function

The linear aggregation function is the same as in the perceptron:

<img src="/assets/post-6/linear-function-adaline.png" width="80%"/>

For a real-valued prediction problem, this is enough. 

### Threshold decision function

When dealing with a *binary classification problem*, we will still use a threshold function, as in the perceptron, by taking the sign of the linear function as: 

$$
\hat{y}' = f(\hat{y}) =
\begin{cases}
+1, & \text{if } \hat{y}\ \text{> 0} \\
-1, & \text{otherwise}
\end{cases}
$$

where $\hat{y}$ is the output of the linear function.

### The Perceptron and ADALINE fundamental difference

If you read [my previous article about the perceptron](https://pabloinsente.github.io/the-perceptron), you may be wondering what's the difference between the perceptron and the ADALINE considering that both end up using a threshold function to make classifications. The difference is the **learning procedure to update the weight** of the network. The perceptron updates the weights by computing the difference between the expected and predicted *class values*. In other words, the perceptron always compares +1 or -1 (predicted values) to +1 or -1 (expected values). An important consequence of this is that perceptron *only learns when errors are made*. In contrast, the ADALINE computes the difference between the expected class value $y$ (+1 or -1), and the *continuous* output value $\hat{y}$ from the linear function, which can be *any real number*. This is crucial because it means the ADALINE can learn *even when no classification mistake has been made*. This is a consequence of the fact that predicted class values $\hat{y}'$ do not influence the error computation. Since the ADALINE learns *all the time* and the perceptron only after errors, the ADALINE will find a solution faster than the perceptron for the same problem. **Figure 1** illustrate this difference in the paths and formulas highlighted in red.

**Figure 1**

<img src="/assets/post-6/adaline-math.png" width="100%"/>

### The ADALINE error surface

Before approaching the formal definition of the ADALINE learning procedure, let's briefly explore what does it mean to "*minimize the mean of the sum of squared errors*". If you are familiar with the [least-squares method](https://en.wikipedia.org/wiki/Least_squares) in regression analysis, this is exactly the same. You can skip to the next section if you feel confident about it.

In a single iteration, the error in the ADALINE is calculated as $(y - \hat{y})^2$, in words, by squaring the difference between the *expected value* and the *predicted value*. This process of comparing the expected and predicted values is repeated for all cases, $j=1$ to $j=n$, in a given dataset. Once we add the squared difference for the entire dataset and divide by the total, we obtained the so-called *mean of squared errors (MSE)*. Formally:

<img src="/assets/post-6/sse.png" width="80%"/>

**Figure 2** shows a visual example of the least-squares method with one predictor. The horizontal axis represents the $x_1$ predictor (or feature), the vertical axis represents the predicted value $\hat{y}$, and the pinkish dots represent the expected values (real data points). If you remember your high-school algebra, you may know that $\hat{y}=w_1b+w_2x_1$ defines a line a cartesian plane. They key, is that the *intercept* (i.e., where the line begins) and the *slope* (i.e., degree of inclination) of the line is determined by the $w_1$ and $w_2$ weights. The $b$ and $x_1$ values are given, *do not change*, therefore, they can't influence the shape of the line.

**Figure 2**

<img src="/assets/post-6/least-squares.png" width="60%"/>

The goal of the least-squares algorithm is to generate as little cumulative error as possible. This equals to find the line that best fit the points in the cartesian plane. Since the weights are the *only values* we can adjust to change the shape of the line, **different pairs of weights will generate different means of squared errors**. This is our gateway to the idea of finding a *minima* in an error surface. Imagine the following: you are trying to find the set of weights, $w_1$ and $w_2$ that would generate the smallest mean of squared error. Your weights can take values ranging from 0 to 1, and your error can go from 0 to 1 (or 0% to 100% thinking proportionally). Now, you decide to plot the mean of squared errors against all possible combinations of $w_1$ and $w_2$. **Figure 3** shows the resulting surface:

**Figure 3**

<img src="/assets/post-6/sse-surface.png" width="100%"/>


We call this an *error surface*. In this case, the shape of the error surface is similar to a cone or pyramid with -crucially- a single point where the error goes all the way down to zero at the bottom of the object. In mathematics, this point is known as **[global minima](https://en.wikipedia.org/wiki/Maxima_and_minima)**. This type of situation, when a unique set of weights defines a single point where the error is zero, is known as a [convex optimization problem](https://en.wikipedia.org/wiki/Convex_optimization). If you try this document in interactive mode (mybinder or locally), you can run the code below and play with the interactive 3D cone (*remove the # symbols and run the cell to see the plot*).


```python
import plotly.graph_objs as go
import numpy as np

x = np.arange(0,1.1,0.1)
y = np.arange(0,1.1,0.1)
z = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1,.8,.8,.8,.8,.8,.8,.8,.8,.8, 1],
    [1,.8,.6,.6,.6,.6,.6,.6,.6,.8, 1],
    [1,.8,.6,.4,.4,.4,.4,.4,.6,.8, 1],
    [1,.8,.6,.4,.2,.2,.2,.4,.6,.8, 1],
    [1,.8,.6,.4,.2,.0,.2,.4,.6,.8, 1],
    [1,.8,.6,.4,.2,.2,.2,.4,.6,.8, 1],
    [1,.8,.6,.4,.4,.4,.4,.4,.6,.8, 1],
    [1,.8,.6,.6,.6,.6,.6,.6,.6,.8, 1],
    [1,.8,.8,.8,.8,.8,.8,.8,.8,.8, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

fig = go.Figure(go.Surface(x=x, y=y, z=z))

scene = dict(xaxis_title='w<sub>1</sub>',
             yaxis_title='w<sub>2</sub>',
             zaxis_title='mean of squared errors',
             aspectratio= {"x": 1, "y": 1, "z": 0.6},
             camera_eye= {"x": 1, "y": -1, "z": 0.5})

####~~~ Uncomment in binder or locally to see 3D plot ~~~~####

# fig.layout.update(scene=scene,
#                    width=700,
#                    margin=dict(r=20, b=10, l=10, t=10))


# fig.show()
```

**All neural networks can be seen as solving optimization problems**, usually, in high-dimensional spaces, with thousands or millions of weights to be adjusted to find the best solution. [AlexNet](https://en.wikipedia.org/wiki/AlexNet), the model that won the ImageNet Visual Recognition Challenge in 20212, has 60,954,656 adjustable parameters. Nowadays, in 2020, most state-of-the-art neural networks have several orders of magnitude more parameters than AlexNet. 

Alas, the bad news is that most problems worth solving in cognitive science are *nonconvex*, meaning that finding the so-called global minima becomes extremely hard, and in most cases can't be guaranteed. In the 3D case, instead of having a nice cone-like error surface, we obtain something more similar to a complex landscape of mountains and valleys, like the Cordillera de Los Andes or the Rocky Mountains. **Figure 4** shows an example of such a landscape:

**Figure 4**

<img src="/assets/post-6/sse-nonconvex.png" width="100%"/>


Now, instead of having a unique point where the error is at its minimum, we have *multiple low points or "valleys" at different sections in the surface*. Those "valleys" are called **[local minima](https://en.wikipedia.org/wiki/Maxima_and_minima)**, or the point of minimum error for that section. Ideally, we always want to find the "global minima", yet, with a landscape like this, finding it may become very hard and slow. If you try this document in interactive mode (mybinder or locally), you can run the code below and play with the interactive 3D surface (*remove the # symbols and run the cell to see the plot*).


```python
import plotly.graph_objs as go
import numpy as np

x = np.arange(0,1.1,0.1)
y = np.arange(0,1.1,0.1)
z = [
    [.7,.6,.6,.5,.7,.4,.8,.3,.3,.3,.5],
    [.4,.4,.4,.4,.8,.5,.4,.2,.2,.2,.3],
    [.3,.3,.3,.3,.3,.5,.1,.1,.1,.2,.3],
    [.3,.2,.2,.2,.3,.4,.4, 0,.1,.2,.3],
    [.3,.2,.1,.2,.3,.2,.2,.3,.4,.7,.7],
    [.3,.2,.2,.2,.3,.5,.5,.5,.5,.4,.7],
    [.3,.3,.3,.3,.3,.2,.2,.2,.2,.4,.7],
    [.2,.2,.2,.3,.4,.4,.2,.1,.2,.4,.7],
    [.2,.3,.1,.3,.6,.6,.2,.2,.2,.4,.7],
    [.2,.3,.3,.3,.8,.8,.4,.4,.4,.5,.8],
    [.3,.5,.5,.5,.9,.9,.8,.6,.8,.8, 1]
]

fig = go.Figure(go.Surface(x=x, y=y, z=z))

scene = dict(xaxis_title='w<sub>1</sub>',
             yaxis_title='w<sub>2</sub>',
             zaxis_title='mean of squared errors',
             aspectratio= {"x": 1, "y": 1, "z": 0.6},
             camera_eye= {"x": 1, "y": -1, "z": 0.5})

####~~~ Uncomment in binder or locally to see 3D plot ~~~~####

# fig.layout.update(scene=scene,
#                   width=700,
#                   margin=dict(r=20, b=10, l=10, t=10))
#
# fig.show()
```

### Learning procedure

By now, we know that we want to find a set of parameters that minimize the mean of squared errors. The ADALINE approaches this by utilizing the so-called **gradient descent algorithm**. For convex problems (cone-like error surface), gradient descent is guaranteed to find the global minima. For nonconvex problems, gradient descent is only guaranteed to find a local minimum, that may or may not be the global minima as well. At this point, we will only discuss convex optimization problems. 

Imagine that you are hiker at the top of a mountain in the side of a valley. Similar to **Figure 5**. Your goal is to reach the base of the valley. Logically, you would want to walk downhill over the hillside until you reach the base. In the context of training neural networks, this is what we call "descending a gradient". Now, it would be nice if you could do this *efficiently*, meaning to follow the path that will get you faster to the base of the valley. In gradient descent terms, this equals to move along the error surface in the direction where the gradient (degree of inclination) is steepest. As a hiker, you can visually inspect your surroundings to determine the best path. In an optimization context, we can use the [chain-rule of calculus](https://en.wikipedia.org/wiki/Chain_rule) to estimate the gradient and adjust the weights.

**Figure 5**

<img src="/assets/post-6/gradient-hiker.png" width="60%"/>


For conciseness, let's define the error of the network as function $E$.  

$$
E(\hat{y}) = \frac{1} n\sum_{j=1}^{n}(y_j - \hat{y_j})^2
$$

If we expand $\hat{y_j}$, whe obtain:

$$
E(w,x) = \frac{1} n\sum_{j=1}^{n}(y_j - (b+\sum_{i}w_ix_i)_j)^2
$$

Now, remember that the only values we can adjust to change $\hat{y}$ are the weights, $w_i$. In differential calculus, taking derivatives means calculating the *rate of change of a function with respect to an infinitely small change in an input argument*. If "infinitely small" sounds like nonsense to you, for practical purposes, think about it as a very small change, let's say, 0.000001. In our case, it means to compute the rate of change of the $E$ function in response to a very small change in $w$. That is what we call to compute a gradient, that we will call $\Delta$, at a point in the error surface. 

Widrow and Hoff have the idea that instead of computing the gradient for the total mean squared error $E$, they could approximate the gradient's value by computing the partial derivative of the error with respect to the weights on each iteration. Since we are dealing with a single case, let's drop the summation symbols and indices for clarity. The function to derivate becomes: 


$$
e(w,x) =(y-(b+wx))^2 
$$

Now comes the fun part. By applying the chain-rule of calculus, the gradient of $e$:

<img src="/assets/post-6/gradient-math.png" width="60%"/>


This may come as a surprise to you, but the gradient, in this case, is as simple as **2 times the difference between the expected and predicted value**. Now we know the $\hat{\Delta}$ we need to update the weights at each iteration. Finally, the rule to update the weights says the following: "**change the weight, $w_j$, by a portion, $\eta$, of the calculated negative gradient, $\Delta_j$**". We use the negative of the gradient because we want to go "downhill", otherwise, you will be climbing the surface in the wrong direction. Formally, this is:

<img src="/assets/post-6/weight-update.png" width="40%"/>


We use a portion ($\eta$) of the gradient instead of the full value to avoid "bouncing around" the minima of the function. This is easier to understand by looking at a simplified example as in **Figure 6**

**Figure 6**

<img src="/assets/post-6/step-size.png" width="60%"/>


In the left pane, the value of $\eta$ is too large to allow the ball to reach the minima of the function so the ball "bounces around" the minima without reaching it. In the right pane, the value of $\eta$ is small enough to allow the ball to reach the minima after a few iterations. Adjusting the step-size or learning rate to find a minimum is usually solved by semi-automatically searching over different values of $\eta$ when training networks. "Why don't use very small values of $\eta$ all the time?" Because there is a trade-off on training time. Smaller values of $\eta$ may help to find the minima, but it will also extend the training time as usually more steps are needed to find the solution.

## Code implementation

We will implement the ADALINE from scratch with `python` and `numpy`. The goal is to understand the perceptron step-by-step execution rather than achieving an elegant implementation. I'll break down each step into functions to ensemble everything at the end. 

The first elements of the ADALINE are essentially the same as in the perceptron. Therefore, we could put those functions in a separate module and call the functions instead of writing them all over again. I'm not doing this to facilitate two things: to refresh the inner workings of the algorithm in code, and to provide with the full description for readers have not read [the previous post](https://pabloinsente.github.io/the-perceptron). If you reviewed the perceptron post already, you may want to skip to the `Training loop - Learning procedure` section.

### Generate vector of random weights


```python
import numpy as np

def random_weights(X, random_state: int):
    '''create vector of random weights
    Parameters
    ----------
    X: 2-dimensional array, shape = [n_samples, n_features]
    Returns
    -------
    w: array, shape = [w_bias + n_features]'''
    rand = np.random.RandomState(random_state)
    w = rand.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
    return w
```

Predictions from the ADALINE are obtained by a [linear combination](https://en.wikipedia.org/wiki/Linear_combination) of features and weights. This is the output of the linear aggregation function. It is common practice begining with a vector of small random weights that would be updated later by the perceptron learning rule.

### Compute net input: linear aggregation function


```python
def net_input(X, w):
    '''Compute net input as dot product'''
    return np.dot(X, w[1:]) + w[0]
```

Here we pass the feature matrix and the previously generated vector of random weights to compute the inner product. Remember that we need to add an extra weight for the bias term at the beginning of the vector (`w[0`)

### Compute predictions: threshold function


```python
def predict(X, w):
    '''Return class label after unit step'''
    return np.where(net_input(X, w) >= 0.0, 1, -1)
```

Remember that although ADALINE learning rule works by comparing the output of a linear function against the class labels when doing predictions, we still need to pass the output by a *threshold function* to get class labels as in the perceptron.

### Training loop - Learning procedure


```python
def fit(X, y, eta=0.001, n_iter=1):
    '''loop over exemplars and update weights'''
    mse_iteration = []
    w = random_weights(X, random_state=1)
    for pair in range(n_iter):
        output = net_input(X, w)
        gradient = 2*(y - output)
        w[1:] += eta*(X.T @ gradient)
        w[0] += eta*gradient.sum()
        mse = (((y - output)**2).sum())/len(y)
        mse_iteration.append(mse)
    return w, mse_iteration
```

Let's examine the fit method that implements the ADALINE learning procedure:

* Create a vector of random weights by using the `random_weights` function with dimensionality equal to the number of columns in the feature matrix
* Loop over the entire dataset `n_iter` times with `for pair in range(n_iter)`
* Compute the inner product between the feature matrix $X$ and the weight vector $w$ by using the `net_input(X, w)` function
* Compute the gradient of error with respect to the weights `2*(y - output)`
* Update the weights in proportion to the learning rate $\eta$ by `w[1:] += eta*(X.T @ gradient)` and `w[0] += eta*gradient.sum()`
* Compute the MSE `mse = (((y - output)**2).sum())/len(y)`
* Save the MSE for each iteration`mse_iteration.append(mse)`

## Application: classification using the peceptron

We will use the same problem as in the perceptron to test the ADALINE: **classifying birds by their weight and wingspan**. I will reproduce the synthetic dataset with two species: *Wandering Albatross* and *Great Horned Owl*. We are doing this to compare the performance of the ADALINE against the perceptron, which hopefully will expand our understanding of these algorithms.

One more time, I'll repeat all the code to set up the classification problem, for the same reasons explained before. If you reviewed the perceptron chapter, you can skip these steps. 

### Generate dataset

Let's first create a function to generate our synthetic data


```python
def species_generator(mu1, sigma1, mu2, sigma2, n_samples, target, seed):
    '''creates [n_samples, 2] array
    
    Parameters
    ----------
    mu1, sigma1: int, shape = [n_samples, 2]
        mean feature-1, standar-dev feature-1
    mu2, sigma2: int, shape = [n_samples, 2]
        mean feature-2, standar-dev feature-2
    n_samples: int, shape= [n_samples, 1]
        number of sample cases
    target: int, shape = [1]
        target value
    seed: int
        random seed for reproducibility
    
    Return
    ------
    X: ndim-array, shape = [n_samples, 2]
        matrix of feature vectors
    y: 1d-vector, shape = [n_samples, 1]
        target vector
    ------
    X'''
    rand = np.random.RandomState(seed)
    f1 = rand.normal(mu1, sigma1, n_samples)
    f2 = rand.normal(mu2, sigma2, n_samples)
    X = np.array([f1, f2])
    X = X.transpose()
    y = np.full((n_samples), target)
    return X, y
```

According to Wikipedia, the [wandering albatross](https://en.wikipedia.org/wiki/Wandering_albatross) mean weight is around 9kg (19.8lbs), and their mean wingspan is around 3m (9.8ft). I will generate a random sample of 100 albatross with the indicated mean values plus some variance.


```python
albatross_weight_mean = 9000 # in grams
albatross_weight_variance =  800 # in grams
albatross_wingspan_mean = 300 # in cm
albatross_wingspan_variance = 20 # in cm 
n_samples = 100
target = 1
seed = 100

# aX: feature matrix (weight, wingspan)
# ay: target value (1)
aX, ay = species_generator(albatross_weight_mean, albatross_weight_variance,
                           albatross_wingspan_mean, albatross_wingspan_variance,
                           n_samples,target,seed )
```


```python
import pandas as pd

albatross_dic = {'weight-(gm)': aX[:,0],
                 'wingspan-(cm)': aX[:,1], 
                 'species': ay,
                 'url': "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}

# put values in a relational table (pandas dataframe)
albatross_df = pd.DataFrame(albatross_dic)
```

According to Wikipedia, the [great horned owl](https://en.wikipedia.org/wiki/Great_horned_owl) mean weight is around 1.2kg (2.7lbs), and its mean wingspan is around 1.2m (3.9ft). Again, I will generate a random sample of 100 owls with the indicated mean values plus some variance. 


```python
owl_weight_mean = 1000 # in grams
owl_weight_variance =  200 # in grams
owl_wingspan_mean = 100 # in cm
owl_wingspan_variance = 15 # in cm
n_samples = 100
target = -1
seed = 100

# oX: feature matrix (weight, wingspan)
# oy: target value (1)
oX, oy = species_generator(owl_weight_mean, owl_weight_variance,
                           owl_wingspan_mean, owl_wingspan_variance,
                           n_samples,target,seed )
```


```python
owl_dic = {'weight-(gm)': oX[:,0],
             'wingspan-(cm)': oX[:,1], 
             'species': oy,
             'url': "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}

# put values in a relational table (pandas dataframe)
owl_df = pd.DataFrame(owl_dic)
```

Now, we concatenate the datasets into a single dataframe.


```python
df = albatross_df.append(owl_df, ignore_index=True)
```

### Plot synthetic dataset

To appreciate the difference in weight and wingspan between albatross and eagles, we can generate a 2-D chart.


```python
import altair as alt

alt.Chart(df).mark_image(
    width=20,
    height=20
).encode(
    x="weight-(gm)",
    y="wingspan-(cm)",
    url="url"
).properties(
    title='Chart 1'
)
```





<div id="altair-viz-d253e542ebb848beabab3fa3cd4c8e4e"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    const outputDiv = document.getElementById("altair-viz-d253e542ebb848beabab3fa3cd4c8e4e");
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.0.2?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function loadScript(lib) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = paths[lib];
        s.async = true;
        s.onload = () => resolve(paths[lib]);
        s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
        document.getElementsByTagName("head")[0].appendChild(s);
      });
    }
    
    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }
    
    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }
    
    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else if (typeof vegaEmbed === "function") {
      displayChart(vegaEmbed);
    } else {
      loadScript("vega")
        .then(() => loadScript("vega-lite"))
        .then(() => loadScript("vega-embed"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-80d394c70723b55f8d9f185712535af2"}, "mark": {"type": "image", "height": 20, "width": 20}, "encoding": {"url": {"type": "nominal", "field": "url"}, "x": {"type": "quantitative", "field": "weight-(gm)"}, "y": {"type": "quantitative", "field": "wingspan-(cm)"}}, "title": "Chart 1", "$schema": "https://vega.github.io/schema/vega-lite/v4.0.2.json", "datasets": {"data-80d394c70723b55f8d9f185712535af2": [{"weight-(gm)": 7600.187621556242, "wingspan-(cm)": 265.90697588478076, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9274.144322662001, "wingspan-(cm)": 277.27477986345275, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9922.428642050916, "wingspan-(cm)": 240.5336905189823, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8798.051170782888, "wingspan-(cm)": 300.6663455627773, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9785.056629560986, "wingspan-(cm)": 295.02222665883784, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9411.375073115505, "wingspan-(cm)": 290.996471299767, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9176.94373537712, "wingspan-(cm)": 302.64855602297547, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8143.965335545366, "wingspan-(cm)": 300.44427856078784, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8848.40333534146, "wingspan-(cm)": 306.34735951882135, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9204.001155418706, "wingspan-(cm)": 284.95171644549924, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8633.5784115979, "wingspan-(cm)": 274.072163856997, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9348.130790498313, "wingspan-(cm)": 301.90278887130904, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8533.123959741868, "wingspan-(cm)": 291.5256980011316, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9653.477657348623, "wingspan-(cm)": 276.2803287014165, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9538.17664456773, "wingspan-(cm)": 292.6907601464675, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8916.471085287498, "wingspan-(cm)": 274.5795391830668, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8574.975698518472, "wingspan-(cm)": 331.7234187684647, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9823.786148106678, "wingspan-(cm)": 313.8678131703318, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8649.491501836466, "wingspan-(cm)": 260.83837531584265, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8105.345402995651, "wingspan-(cm)": 297.3039737602001, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 10295.185328540207, "wingspan-(cm)": 269.1876795089477, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 10233.284139610725, "wingspan-(cm)": 340.93427936964275, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8798.49668862943, "wingspan-(cm)": 272.0600131009343, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8326.051409398962, "wingspan-(cm)": 278.05656030720354, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9147.614952451155, "wingspan-(cm)": 295.2257426137064, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9749.665760887161, "wingspan-(cm)": 271.41866203103416, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9584.800275067846, "wingspan-(cm)": 318.9800955301052, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 10089.244900116264, "wingspan-(cm)": 299.6120482807507, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8739.009552638157, "wingspan-(cm)": 317.8919541152003, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9044.540811883822, "wingspan-(cm)": 315.1938623970041, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9177.919686844243, "wingspan-(cm)": 270.0455923783366, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7845.426403819731, "wingspan-(cm)": 276.12228046416124, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8394.918155524452, "wingspan-(cm)": 325.92525172798116, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9653.16320881543, "wingspan-(cm)": 319.0455125216378, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9600.355809227343, "wingspan-(cm)": 275.6549173871797, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8635.242458025597, "wingspan-(cm)": 296.85469665249724, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9951.6978144233, "wingspan-(cm)": 269.84829679471227, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7647.506538893116, "wingspan-(cm)": 302.15768261613226, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7914.88076091095, "wingspan-(cm)": 314.941113101983, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8014.052388868059, "wingspan-(cm)": 308.5935287172522, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8564.448670662028, "wingspan-(cm)": 271.6991415829495, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8465.462610549257, "wingspan-(cm)": 287.18480153978857, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9005.851650583123, "wingspan-(cm)": 315.5925260732739, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8509.64901161747, "wingspan-(cm)": 291.23758167302316, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 10039.798459804248, "wingspan-(cm)": 341.49586335893133, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7613.523501077375, "wingspan-(cm)": 293.1340463563506, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8213.351920696296, "wingspan-(cm)": 287.6674125663361, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9286.006202533892, "wingspan-(cm)": 315.2636729211998, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7709.13719774226, "wingspan-(cm)": 303.85834383646613, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 10176.571093289704, "wingspan-(cm)": 293.0308213869526, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8049.585922145825, "wingspan-(cm)": 345.9730788142735, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8560.20304517161, "wingspan-(cm)": 296.69580894718536, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8247.963070764185, "wingspan-(cm)": 309.3259873671438, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8337.654108507304, "wingspan-(cm)": 305.3997447726218, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9087.090774266944, "wingspan-(cm)": 293.6033790576382, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9406.247672393856, "wingspan-(cm)": 277.04516800246824, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8310.218122791615, "wingspan-(cm)": 334.0724797624141, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9999.575794181583, "wingspan-(cm)": 285.5569845988849, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8936.31100326608, "wingspan-(cm)": 321.87373299317443, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8288.214814987972, "wingspan-(cm)": 295.4096449352009, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8294.561288413586, "wingspan-(cm)": 299.8220267341578, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9014.911159590449, "wingspan-(cm)": 289.1360398318566, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9190.275697538897, "wingspan-(cm)": 315.06124375383956, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9010.83883890289, "wingspan-(cm)": 267.8112220765409, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7691.576480495342, "wingspan-(cm)": 338.8652452686799, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8164.632097832547, "wingspan-(cm)": 271.0512777536083, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9490.431105350037, "wingspan-(cm)": 302.60496910705405, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9588.96417065906, "wingspan-(cm)": 318.9872172932197, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9821.537151519833, "wingspan-(cm)": 259.69622565754946, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7854.247511152858, "wingspan-(cm)": 298.40918826131775, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7527.049359850626, "wingspan-(cm)": 306.0209892757613, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9292.874580933843, "wingspan-(cm)": 266.3020007662964, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8734.578291957752, "wingspan-(cm)": 304.447816188909, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8448.625617528198, "wingspan-(cm)": 286.3015652950554, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 10627.686049203947, "wingspan-(cm)": 297.4759763257285, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8559.428470468325, "wingspan-(cm)": 339.8054729950818, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9600.362664261473, "wingspan-(cm)": 310.459956090415, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7954.406128735345, "wingspan-(cm)": 299.6730919448503, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9464.45866863542, "wingspan-(cm)": 291.683673283187, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8116.381525870165, "wingspan-(cm)": 272.829941264804, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9552.097176179766, "wingspan-(cm)": 289.7114021726243, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9549.512052910724, "wingspan-(cm)": 295.6787975999348, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7746.649976337287, "wingspan-(cm)": 308.4476044084396, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9723.979297173344, "wingspan-(cm)": 278.1191413793553, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9623.057919458453, "wingspan-(cm)": 324.7381577038045, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9342.586296477393, "wingspan-(cm)": 295.39430643145784, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9087.097591918331, "wingspan-(cm)": 285.91163600529956, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9022.62690785846, "wingspan-(cm)": 288.17249757829654, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8536.93934016721, "wingspan-(cm)": 314.73990338036424, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8040.439040644855, "wingspan-(cm)": 308.7173450502982, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7635.238395409464, "wingspan-(cm)": 335.51987171013536, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9295.331165685604, "wingspan-(cm)": 310.2614875767929, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 10501.258741569733, "wingspan-(cm)": 323.4105396589629, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8698.47731986482, "wingspan-(cm)": 341.5542446450041, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 10465.548865460429, "wingspan-(cm)": 290.8815596157196, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9002.413947224972, "wingspan-(cm)": 312.98345854509364, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8939.181227420302, "wingspan-(cm)": 296.50436891097, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9003.16607519008, "wingspan-(cm)": 320.3452868650234, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8851.988711282309, "wingspan-(cm)": 288.00033910302267, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7010.278771821784, "wingspan-(cm)": 331.5233344863841, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 650.0469053890605, "wingspan-(cm)": 74.43023191358557, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1068.5360806655003, "wingspan-(cm)": 82.95608489758956, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1230.6071605127288, "wingspan-(cm)": 55.400267889236716, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 949.512792695722, "wingspan-(cm)": 100.49975917208295, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1196.2641573902463, "wingspan-(cm)": 96.26666999412838, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1102.8437682788765, "wingspan-(cm)": 93.24735347482523, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1044.23593384428, "wingspan-(cm)": 101.9864170172316, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 785.9913338863414, "wingspan-(cm)": 100.33320892059086, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 962.1008338353649, "wingspan-(cm)": 104.76051963911601, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1051.0002888546762, "wingspan-(cm)": 88.71378733412445, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 908.3946028994751, "wingspan-(cm)": 80.55412289274774, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1087.0326976245785, "wingspan-(cm)": 101.4270916534818, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 883.280989935467, "wingspan-(cm)": 93.6442735008487, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1163.3694143371558, "wingspan-(cm)": 82.2102465260624, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1134.5441611419321, "wingspan-(cm)": 94.5180701098506, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 979.1177713218744, "wingspan-(cm)": 80.93465438730009, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 893.743924629618, "wingspan-(cm)": 123.79256407634853, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1205.9465370266691, "wingspan-(cm)": 110.40085987774883, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 912.3728754591166, "wingspan-(cm)": 70.628781486882, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 776.3363507489128, "wingspan-(cm)": 97.97798032015008, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1323.796332135052, "wingspan-(cm)": 76.8907596317108, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1308.3210349026813, "wingspan-(cm)": 130.70070952723208, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 949.6241721573574, "wingspan-(cm)": 79.04500982570075, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 831.5128523497405, "wingspan-(cm)": 83.54242023040266, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1036.9037381127885, "wingspan-(cm)": 96.41930696027981, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1187.4164402217905, "wingspan-(cm)": 78.56399652327563, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1146.2000687669617, "wingspan-(cm)": 114.23507164757889, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1272.3112250290662, "wingspan-(cm)": 99.70903621056303, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 934.7523881595395, "wingspan-(cm)": 113.4189655864002, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1011.1352029709553, "wingspan-(cm)": 111.39539679775308, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1044.479921711061, "wingspan-(cm)": 77.53419428375246, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 711.3566009549327, "wingspan-(cm)": 82.09171034812093, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 848.729538881113, "wingspan-(cm)": 119.44393879598586, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1163.290802203857, "wingspan-(cm)": 114.28413439122836, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1150.0889523068356, "wingspan-(cm)": 81.74118804038477, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 908.8106145063996, "wingspan-(cm)": 97.64102248937293, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1237.9244536058252, "wingspan-(cm)": 77.3862225960342, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 661.8766347232792, "wingspan-(cm)": 101.6182619620992, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 728.7201902277375, "wingspan-(cm)": 111.2058348264872, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 753.5130972170148, "wingspan-(cm)": 106.44514653793914, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 891.1121676655072, "wingspan-(cm)": 78.7743561872121, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 866.3656526373145, "wingspan-(cm)": 90.38860115484142, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1001.4629126457806, "wingspan-(cm)": 111.69439455495544, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 877.4122529043675, "wingspan-(cm)": 93.42818625476735, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1259.949614951062, "wingspan-(cm)": 131.12189751919848, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 653.3808752693437, "wingspan-(cm)": 94.85053476726296, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 803.337980174074, "wingspan-(cm)": 90.75055942475208, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1071.501550633473, "wingspan-(cm)": 111.44775469089988, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 677.2842994355649, "wingspan-(cm)": 102.8937578773496, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1294.142773322426, "wingspan-(cm)": 94.77311604021445, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 762.3964805364561, "wingspan-(cm)": 134.47980911070513, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 890.050761292902, "wingspan-(cm)": 97.52185671038902, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 811.9907676910464, "wingspan-(cm)": 106.99449052535785, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 834.4135271268258, "wingspan-(cm)": 104.04980857946633, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1021.7726935667359, "wingspan-(cm)": 95.20253429322868, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1101.561918098464, "wingspan-(cm)": 82.78387600185121, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 827.554530697904, "wingspan-(cm)": 125.55435982181058, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1249.8939485453957, "wingspan-(cm)": 89.16773844916369, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 984.0777508165202, "wingspan-(cm)": 116.40529974488082, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 822.0537037469933, "wingspan-(cm)": 96.55723370140066, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 823.6403221033966, "wingspan-(cm)": 99.86652005061835, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1003.7277898976121, "wingspan-(cm)": 91.85202987389242, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1047.5689243847244, "wingspan-(cm)": 111.29593281537969, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1002.7097097257225, "wingspan-(cm)": 75.85841655740569, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 672.8941201238356, "wingspan-(cm)": 129.14893395150995, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 791.1580244581367, "wingspan-(cm)": 78.28845831520623, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1122.6077763375092, "wingspan-(cm)": 101.95372683029053, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1147.2410426647648, "wingspan-(cm)": 114.2404129699148, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1205.384287879958, "wingspan-(cm)": 69.7721692431621, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 713.5618777882146, "wingspan-(cm)": 98.80689119598833, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 631.7623399626566, "wingspan-(cm)": 104.51574195682099, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1073.2186452334608, "wingspan-(cm)": 74.7265005747223, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 933.6445729894383, "wingspan-(cm)": 103.33586214168173, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 862.1564043820493, "wingspan-(cm)": 89.72617397129154, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1406.9215123009867, "wingspan-(cm)": 98.10698224429635, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 889.8571176170815, "wingspan-(cm)": 129.85410474631135, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1150.0906660653682, "wingspan-(cm)": 107.84496706781127, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 738.6015321838362, "wingspan-(cm)": 99.7548189586377, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1116.114667158855, "wingspan-(cm)": 93.76275496239025, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 779.0953814675413, "wingspan-(cm)": 79.622455948603, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1138.0242940449416, "wingspan-(cm)": 92.28355162946822, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1137.3780132276809, "wingspan-(cm)": 96.75909819995108, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 686.6624940843218, "wingspan-(cm)": 106.33570330632968, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1180.9948242933363, "wingspan-(cm)": 83.58935603451647, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1155.7644798646133, "wingspan-(cm)": 118.55361827785339, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1085.646574119348, "wingspan-(cm)": 96.54572982359338, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1021.7743979795829, "wingspan-(cm)": 89.43372700397465, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1005.6567269646147, "wingspan-(cm)": 91.12937318372242, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 884.2348350418023, "wingspan-(cm)": 111.05492753527318, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 760.1097601612138, "wingspan-(cm)": 106.53800878772365, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 658.809598852366, "wingspan-(cm)": 126.6399037826015, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1073.8327914214012, "wingspan-(cm)": 107.69611568259468, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1375.3146853924331, "wingspan-(cm)": 117.55790474422216, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 924.619329966205, "wingspan-(cm)": 131.16568348375304, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1366.3872163651072, "wingspan-(cm)": 93.1611697117897, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1000.6034868062428, "wingspan-(cm)": 109.73759390882023, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 984.7953068550753, "wingspan-(cm)": 97.37827668322753, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 1000.7915187975199, "wingspan-(cm)": 115.25896514876754, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 962.9971778205772, "wingspan-(cm)": 91.000254327267, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}, {"weight-(gm)": 502.5696929554461, "wingspan-(cm)": 123.64250086478809, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/owl.png"}]}}, {"mode": "vega-lite"});
</script>



From **Chart 1** is clear that the albatross is considerably larger than the owls, therefore the ADALINE should be able to find a plane to separate the data relatively fast.

### ADALINE training

Before training the ADALINE, we will shuffle the rows in the dataset. This is not technically necessary, but it would help the ADALINE to converge faster.


```python
df_shuffle = df.sample(frac=1, random_state=1).reset_index(drop=True)
X = df_shuffle[['weight-(gm)','wingspan-(cm)']].to_numpy()
y = df_shuffle['species'].to_numpy()
```

We use the fit function, with a learning rate or $\eta$ of 1e-10, and run 12 iterations of the algorithm. On each iteration, the entire dataset is passed by the ADALINE once. 


```python
w, mse = fit(X, y, eta=1e-10, n_iter=12)
```


```python
y_pred = predict(X, w)
num_correct_predictions = (y_pred == y).sum()
accuracy = (num_correct_predictions / y.shape[0]) * 100
print('ADALINE accuracy: %.2f%%' % accuracy)
```

    ADALINE accuracy: 99.50%


After 12 runs, the ADALINE reached 99.5% accuracy, which is the *same accuracy that the perceptron achieved with 200 runs given the same data* (see "Perceptron training" section in [the perceptron post](https://pabloinsente.github.io/the-perceptron)). A massive reduction in training time. As I mentioned before, this is related to the fact that the ADALINE learns (i.e., update the weights) after each iteration, instead of only after makes a classification mistake as the perceptron.

Examining the reduction in mean-squared-error at each time-step (**Chart 2**) reveals a trajectory where the error goes down really fast in the first few iterations, and then slow down as it approaches zero.


```python
mse_df = pd.DataFrame({'mse':mse, 'time-step': np.arange(0, len(mse))})
base = alt.Chart(mse_df).encode(x="time-step:O")
base.mark_line().encode(
    y="mse"
).properties(
    width=400,
    title='Chart 2'
)
```





<div id="altair-viz-a4512463e8574185a9b8971188f15d31"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    const outputDiv = document.getElementById("altair-viz-a4512463e8574185a9b8971188f15d31");
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.0.2?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function loadScript(lib) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = paths[lib];
        s.async = true;
        s.onload = () => resolve(paths[lib]);
        s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
        document.getElementsByTagName("head")[0].appendChild(s);
      });
    }
    
    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }
    
    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }
    
    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else if (typeof vegaEmbed === "function") {
      displayChart(vegaEmbed);
    } else {
      loadScript("vega")
        .then(() => loadScript("vega-lite"))
        .then(() => loadScript("vega-embed"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-6f8944e207a49984b51dc6c79395c39f"}, "mark": "line", "encoding": {"x": {"type": "ordinal", "field": "time-step"}, "y": {"type": "quantitative", "field": "mse"}}, "title": "Chart 2", "width": 400, "$schema": "https://vega.github.io/schema/vega-lite/v4.0.2.json", "datasets": {"data-6f8944e207a49984b51dc6c79395c39f": [{"mse": 1657.7059195090953, "time-step": 0}, {"mse": 645.6347772640765, "time-step": 1}, {"mse": 251.58682683094383, "time-step": 2}, {"mse": 98.16500374231856, "time-step": 3}, {"mse": 38.43049519101968, "time-step": 4}, {"mse": 15.172960286729813, "time-step": 5}, {"mse": 6.1176641091291435, "time-step": 6}, {"mse": 2.591981953502288, "time-step": 7}, {"mse": 1.2192442640539063, "time-step": 8}, {"mse": 0.6847513186984904, "time-step": 9}, {"mse": 0.47662733844914024, "time-step": 10}, {"mse": 0.39557448463381123, "time-step": 11}]}}, {"mode": "vega-lite"});
</script>



## ADALINE limitations

Although the ADALINE introduced a better training procedure, it did not fix the so-called linear separability constraint problem, the main limitation of the perceptron. Its training procedure it's also vulnerable to what sometimes is informally called "error explosion" or "gradient explosion". We will examine these two problems next.


### Error and gradient explosion

You may have noticed that the learning rate and iterations choice was pretty arbitrary. This is essentially unavoidable in the context of training neural networks with gradient descent methods. Nowadays, there are several "tricks" that can be applied to search for parameters like the learning rate $\eta$ more efficiently, but the problem of searching for those kinds of parameters persist. 

Now, a $\eta$=1e-10 (0.0000000001) looks very small. Let's try with a slightly larger $\eta$=1e-9 and test the ADALINE again.


```python
w_2, mse_2 = fit(X, y, eta=1e-9, n_iter=12)
```


```python
y_pred = predict(X, w_2)
num_correct_predictions = (y_pred == y).sum()
accuracy = (num_correct_predictions / y.shape[0]) * 100
print('Perceptron accuracy: %.2f%%' % accuracy)
```

    Perceptron accuracy: 50.00%



```python
mse_df["mse-2"] = mse_2
```


```python
base = alt.Chart(mse_df).encode(x="time-step:O")
line1 = base.mark_line().encode(y="mse")
line2 = base.mark_line(color='red').encode(y="mse-2")
line1 | line2 | (line1 + line2)
```





<div id="altair-viz-df79ccd6e7b04e70a121d5dbe73ee4da"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    const outputDiv = document.getElementById("altair-viz-df79ccd6e7b04e70a121d5dbe73ee4da");
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.0.2?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function loadScript(lib) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = paths[lib];
        s.async = true;
        s.onload = () => resolve(paths[lib]);
        s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
        document.getElementsByTagName("head")[0].appendChild(s);
      });
    }
    
    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }
    
    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }
    
    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else if (typeof vegaEmbed === "function") {
      displayChart(vegaEmbed);
    } else {
      loadScript("vega")
        .then(() => loadScript("vega-lite"))
        .then(() => loadScript("vega-embed"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "hconcat": [{"mark": "line", "encoding": {"x": {"type": "ordinal", "field": "time-step"}, "y": {"type": "quantitative", "field": "mse"}}}, {"mark": {"type": "line", "color": "red"}, "encoding": {"x": {"type": "ordinal", "field": "time-step"}, "y": {"type": "quantitative", "field": "mse-2"}}}, {"layer": [{"mark": "line", "encoding": {"x": {"type": "ordinal", "field": "time-step"}, "y": {"type": "quantitative", "field": "mse"}}}, {"mark": {"type": "line", "color": "red"}, "encoding": {"x": {"type": "ordinal", "field": "time-step"}, "y": {"type": "quantitative", "field": "mse-2"}}}]}], "data": {"name": "data-707d8a88af07458e2a4b3ad5a98256e6"}, "$schema": "https://vega.github.io/schema/vega-lite/v4.0.2.json", "datasets": {"data-707d8a88af07458e2a4b3ad5a98256e6": [{"mse": 1657.7059195090953, "time-step": 0, "mse-2": 1657.7059195090953}, {"mse": 645.6347772640765, "time-step": 1, "mse-2": 384923.86858839454}, {"mse": 251.58682683094383, "time-step": 2, "mse-2": 89398787.1664703}, {"mse": 98.16500374231856, "time-step": 3, "mse-2": 20762937503.05355}, {"mse": 38.43049519101968, "time-step": 4, "mse-2": 4822208321414.161}, {"mse": 15.172960286729813, "time-step": 5, "mse-2": 1119961618729396.1}, {"mse": 6.1176641091291435, "time-step": 6, "mse-2": 2.6011195365761754e+17}, {"mse": 2.591981953502288, "time-step": 7, "mse-2": 6.041120276277085e+19}, {"mse": 1.2192442640539063, "time-step": 8, "mse-2": 1.4030548646174208e+22}, {"mse": 0.6847513186984904, "time-step": 9, "mse-2": 3.258605793460144e+24}, {"mse": 0.47662733844914024, "time-step": 10, "mse-2": 7.568137201867329e+26}, {"mse": 0.39557448463381123, "time-step": 11, "mse-2": 1.7577057286659103e+29}]}}, {"mode": "vega-lite"});
</script>


The results now look like a mess. The middle pane shows how the MSE error explodes almost immediately. The right pane is there just to show how absurd is the difference between the two $\eta$  values.
The network went from solving the problem in 12 iterations to an MSE of 4822,208,321,414 in the 5th iteration, and to a prediction accuracy equivalent to random guessing. Actually, if you try increasing the learning rate even more, the MSE would be so large it would "overflow" or exceed the capacity of your computer to represent integers. 

This is very strange. What is going on? In simple terms, the step-size is so large that after the first iteration, the error is so far-off from the minima of the function, that it can't find its way in the right direction after that. Instead, the error starts to spiral out of control until it blows up and your computer runs out of memory. Remember, the ADALINE computes the $\hat{y}$ by multiplying each feature by the weights. Consider a bird with $x_1 = 10,000$ and $x_2 = 300$. Let's compute the predicted value for that case:

$$
\hat{y} = 10,000*0.01 + 0.01*300 + 0.01*1 = 103.01
$$

Considering that now the error is computed as:

$$
error = (1 - 103.01)^2 = 10,406
$$

That's a huge number. With enough data, the network should learn to reduce the $w_i$ until are small enough to make sensible predictions. But, if you start with predictions that are too far-off, the network may become unable to get back on track. **Figure 7** succinctly illustrate this idea.  

**Figure 7**  

<img src="/assets/post-6/error-explotion.png" width="60%"/>

### Linear separability constraint

We explored the linear separability constraint in detail in the perceptron Chapter, therefore we will explore this more briefly here. Let's generate a new dataset, with Albatross and Condors. 


```python
condor_weight_mean = 12000 # in grams
condor_weight_variance = 1000 # in grams
condor_wingspan_mean = 290 # in cm
condor_wingspan_variance = 15 # in cm 
n_samples = 100
target = -1
seed = 100

# cX: feature matrix (weight, wingspan)
# cy: target value (1)
cX, cy = species_generator(condor_weight_mean, condor_weight_variance,
                           condor_wingspan_mean, condor_wingspan_variance,
                           n_samples,target,seed )

condor_dic = {'weight-(gm)': cX[:,0],
             'wingspan-(cm)': cX[:,1], 
             'species': cy,
             'url': "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}

# put values in a relational table (pandas dataframe)
condor_df = pd.DataFrame(condor_dic)

df2 = albatross_df.append(condor_df, ignore_index=True)
```


```python
alt.Chart(df2).mark_image(
    width=20,
    height=20
).encode(
    alt.X("weight-(gm)", scale=alt.Scale(domain=(6000, 16000))),
    alt.Y("wingspan-(cm)", scale=alt.Scale(domain=(220, 360))),
    url="url"
).properties(
title="Chart 3"
)
```





<div id="altair-viz-d7aba5bef7884ffaaa0384e805714474"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    const outputDiv = document.getElementById("altair-viz-d7aba5bef7884ffaaa0384e805714474");
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.0.2?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function loadScript(lib) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = paths[lib];
        s.async = true;
        s.onload = () => resolve(paths[lib]);
        s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
        document.getElementsByTagName("head")[0].appendChild(s);
      });
    }
    
    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }
    
    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }
    
    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else if (typeof vegaEmbed === "function") {
      displayChart(vegaEmbed);
    } else {
      loadScript("vega")
        .then(() => loadScript("vega-lite"))
        .then(() => loadScript("vega-embed"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-78ea670c68e7637c818e0cb513b4ec83"}, "mark": {"type": "image", "height": 20, "width": 20}, "encoding": {"url": {"type": "nominal", "field": "url"}, "x": {"type": "quantitative", "field": "weight-(gm)", "scale": {"domain": [6000, 16000]}}, "y": {"type": "quantitative", "field": "wingspan-(cm)", "scale": {"domain": [220, 360]}}}, "title": "Chart 3", "$schema": "https://vega.github.io/schema/vega-lite/v4.0.2.json", "datasets": {"data-78ea670c68e7637c818e0cb513b4ec83": [{"weight-(gm)": 7600.187621556242, "wingspan-(cm)": 265.90697588478076, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9274.144322662001, "wingspan-(cm)": 277.27477986345275, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9922.428642050916, "wingspan-(cm)": 240.5336905189823, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8798.051170782888, "wingspan-(cm)": 300.6663455627773, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9785.056629560986, "wingspan-(cm)": 295.02222665883784, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9411.375073115505, "wingspan-(cm)": 290.996471299767, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9176.94373537712, "wingspan-(cm)": 302.64855602297547, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8143.965335545366, "wingspan-(cm)": 300.44427856078784, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8848.40333534146, "wingspan-(cm)": 306.34735951882135, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9204.001155418706, "wingspan-(cm)": 284.95171644549924, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8633.5784115979, "wingspan-(cm)": 274.072163856997, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9348.130790498313, "wingspan-(cm)": 301.90278887130904, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8533.123959741868, "wingspan-(cm)": 291.5256980011316, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9653.477657348623, "wingspan-(cm)": 276.2803287014165, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9538.17664456773, "wingspan-(cm)": 292.6907601464675, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8916.471085287498, "wingspan-(cm)": 274.5795391830668, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8574.975698518472, "wingspan-(cm)": 331.7234187684647, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9823.786148106678, "wingspan-(cm)": 313.8678131703318, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8649.491501836466, "wingspan-(cm)": 260.83837531584265, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8105.345402995651, "wingspan-(cm)": 297.3039737602001, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 10295.185328540207, "wingspan-(cm)": 269.1876795089477, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 10233.284139610725, "wingspan-(cm)": 340.93427936964275, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8798.49668862943, "wingspan-(cm)": 272.0600131009343, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8326.051409398962, "wingspan-(cm)": 278.05656030720354, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9147.614952451155, "wingspan-(cm)": 295.2257426137064, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9749.665760887161, "wingspan-(cm)": 271.41866203103416, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9584.800275067846, "wingspan-(cm)": 318.9800955301052, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 10089.244900116264, "wingspan-(cm)": 299.6120482807507, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8739.009552638157, "wingspan-(cm)": 317.8919541152003, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9044.540811883822, "wingspan-(cm)": 315.1938623970041, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9177.919686844243, "wingspan-(cm)": 270.0455923783366, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7845.426403819731, "wingspan-(cm)": 276.12228046416124, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8394.918155524452, "wingspan-(cm)": 325.92525172798116, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9653.16320881543, "wingspan-(cm)": 319.0455125216378, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9600.355809227343, "wingspan-(cm)": 275.6549173871797, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8635.242458025597, "wingspan-(cm)": 296.85469665249724, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9951.6978144233, "wingspan-(cm)": 269.84829679471227, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7647.506538893116, "wingspan-(cm)": 302.15768261613226, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7914.88076091095, "wingspan-(cm)": 314.941113101983, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8014.052388868059, "wingspan-(cm)": 308.5935287172522, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8564.448670662028, "wingspan-(cm)": 271.6991415829495, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8465.462610549257, "wingspan-(cm)": 287.18480153978857, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9005.851650583123, "wingspan-(cm)": 315.5925260732739, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8509.64901161747, "wingspan-(cm)": 291.23758167302316, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 10039.798459804248, "wingspan-(cm)": 341.49586335893133, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7613.523501077375, "wingspan-(cm)": 293.1340463563506, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8213.351920696296, "wingspan-(cm)": 287.6674125663361, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9286.006202533892, "wingspan-(cm)": 315.2636729211998, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7709.13719774226, "wingspan-(cm)": 303.85834383646613, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 10176.571093289704, "wingspan-(cm)": 293.0308213869526, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8049.585922145825, "wingspan-(cm)": 345.9730788142735, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8560.20304517161, "wingspan-(cm)": 296.69580894718536, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8247.963070764185, "wingspan-(cm)": 309.3259873671438, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8337.654108507304, "wingspan-(cm)": 305.3997447726218, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9087.090774266944, "wingspan-(cm)": 293.6033790576382, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9406.247672393856, "wingspan-(cm)": 277.04516800246824, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8310.218122791615, "wingspan-(cm)": 334.0724797624141, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9999.575794181583, "wingspan-(cm)": 285.5569845988849, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8936.31100326608, "wingspan-(cm)": 321.87373299317443, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8288.214814987972, "wingspan-(cm)": 295.4096449352009, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8294.561288413586, "wingspan-(cm)": 299.8220267341578, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9014.911159590449, "wingspan-(cm)": 289.1360398318566, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9190.275697538897, "wingspan-(cm)": 315.06124375383956, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9010.83883890289, "wingspan-(cm)": 267.8112220765409, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7691.576480495342, "wingspan-(cm)": 338.8652452686799, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8164.632097832547, "wingspan-(cm)": 271.0512777536083, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9490.431105350037, "wingspan-(cm)": 302.60496910705405, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9588.96417065906, "wingspan-(cm)": 318.9872172932197, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9821.537151519833, "wingspan-(cm)": 259.69622565754946, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7854.247511152858, "wingspan-(cm)": 298.40918826131775, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7527.049359850626, "wingspan-(cm)": 306.0209892757613, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9292.874580933843, "wingspan-(cm)": 266.3020007662964, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8734.578291957752, "wingspan-(cm)": 304.447816188909, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8448.625617528198, "wingspan-(cm)": 286.3015652950554, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 10627.686049203947, "wingspan-(cm)": 297.4759763257285, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8559.428470468325, "wingspan-(cm)": 339.8054729950818, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9600.362664261473, "wingspan-(cm)": 310.459956090415, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7954.406128735345, "wingspan-(cm)": 299.6730919448503, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9464.45866863542, "wingspan-(cm)": 291.683673283187, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8116.381525870165, "wingspan-(cm)": 272.829941264804, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9552.097176179766, "wingspan-(cm)": 289.7114021726243, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9549.512052910724, "wingspan-(cm)": 295.6787975999348, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7746.649976337287, "wingspan-(cm)": 308.4476044084396, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9723.979297173344, "wingspan-(cm)": 278.1191413793553, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9623.057919458453, "wingspan-(cm)": 324.7381577038045, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9342.586296477393, "wingspan-(cm)": 295.39430643145784, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9087.097591918331, "wingspan-(cm)": 285.91163600529956, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9022.62690785846, "wingspan-(cm)": 288.17249757829654, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8536.93934016721, "wingspan-(cm)": 314.73990338036424, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8040.439040644855, "wingspan-(cm)": 308.7173450502982, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7635.238395409464, "wingspan-(cm)": 335.51987171013536, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9295.331165685604, "wingspan-(cm)": 310.2614875767929, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 10501.258741569733, "wingspan-(cm)": 323.4105396589629, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8698.47731986482, "wingspan-(cm)": 341.5542446450041, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 10465.548865460429, "wingspan-(cm)": 290.8815596157196, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9002.413947224972, "wingspan-(cm)": 312.98345854509364, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8939.181227420302, "wingspan-(cm)": 296.50436891097, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 9003.16607519008, "wingspan-(cm)": 320.3452868650234, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 8851.988711282309, "wingspan-(cm)": 288.00033910302267, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 7010.278771821784, "wingspan-(cm)": 331.5233344863841, "species": 1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/albatross.png"}, {"weight-(gm)": 10250.234526945304, "wingspan-(cm)": 264.43023191358554, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12342.680403327502, "wingspan-(cm)": 272.95608489758956, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 13153.035802563643, "wingspan-(cm)": 245.40026788923672, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11747.56396347861, "wingspan-(cm)": 290.4997591720829, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12981.320786951232, "wingspan-(cm)": 286.2666699941284, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12514.218841394382, "wingspan-(cm)": 283.2473534748252, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12221.179669221401, "wingspan-(cm)": 291.9864170172316, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 10929.956669431707, "wingspan-(cm)": 290.3332089205909, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11810.504169176824, "wingspan-(cm)": 294.760519639116, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12255.001444273381, "wingspan-(cm)": 278.71378733412445, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11541.973014497376, "wingspan-(cm)": 270.5541228927477, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12435.163488122893, "wingspan-(cm)": 291.4270916534818, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11416.404949677335, "wingspan-(cm)": 283.64427350084867, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12816.84707168578, "wingspan-(cm)": 272.2102465260624, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12672.720805709661, "wingspan-(cm)": 284.5180701098506, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11895.588856609373, "wingspan-(cm)": 270.9346543873001, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11468.71962314809, "wingspan-(cm)": 313.7925640763485, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 13029.732685133346, "wingspan-(cm)": 300.4008598777488, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11561.864377295582, "wingspan-(cm)": 260.628781486882, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 10881.681753744564, "wingspan-(cm)": 287.9779803201501, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 13618.98166067526, "wingspan-(cm)": 266.8907596317108, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 13541.605174513406, "wingspan-(cm)": 320.7007095272321, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11748.120860786787, "wingspan-(cm)": 269.04500982570073, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11157.564261748703, "wingspan-(cm)": 273.5424202304027, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12184.518690563942, "wingspan-(cm)": 286.4193069602798, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12937.082201108953, "wingspan-(cm)": 268.56399652327565, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12731.000343834809, "wingspan-(cm)": 304.2350716475789, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 13361.556125145331, "wingspan-(cm)": 289.709036210563, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11673.761940797698, "wingspan-(cm)": 303.4189655864002, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12055.676014854776, "wingspan-(cm)": 301.3953967977531, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12222.399608555304, "wingspan-(cm)": 267.53419428375247, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 10556.783004774663, "wingspan-(cm)": 272.09171034812096, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11243.647694405565, "wingspan-(cm)": 309.44393879598584, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12816.454011019287, "wingspan-(cm)": 304.28413439122835, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12750.444761534178, "wingspan-(cm)": 271.74118804038477, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11544.053072531999, "wingspan-(cm)": 287.6410224893729, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 13189.622268029125, "wingspan-(cm)": 267.3862225960342, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 10309.383173616396, "wingspan-(cm)": 291.6182619620992, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 10643.600951138687, "wingspan-(cm)": 301.2058348264872, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 10767.565486085074, "wingspan-(cm)": 296.44514653793914, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11455.560838327536, "wingspan-(cm)": 268.7743561872121, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11331.828263186573, "wingspan-(cm)": 280.3886011548414, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12007.314563228903, "wingspan-(cm)": 301.69439455495547, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11387.061264521837, "wingspan-(cm)": 283.42818625476735, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 13299.748074755309, "wingspan-(cm)": 321.12189751919846, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 10266.90437634672, "wingspan-(cm)": 284.85053476726296, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11016.68990087037, "wingspan-(cm)": 280.75055942475205, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12357.507753167365, "wingspan-(cm)": 301.4477546908999, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 10386.421497177824, "wingspan-(cm)": 292.89375787734957, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 13470.713866612128, "wingspan-(cm)": 284.77311604021446, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 10811.98240268228, "wingspan-(cm)": 324.4798091107051, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11450.25380646451, "wingspan-(cm)": 287.52185671038905, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11059.953838455232, "wingspan-(cm)": 296.99449052535783, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11172.067635634128, "wingspan-(cm)": 294.04980857946634, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12108.86346783368, "wingspan-(cm)": 285.20253429322867, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12507.80959049232, "wingspan-(cm)": 272.7838760018512, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11137.772653489521, "wingspan-(cm)": 315.5543598218106, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 13249.469742726978, "wingspan-(cm)": 279.1677384491637, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11920.3887540826, "wingspan-(cm)": 306.40529974488084, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11110.268518734967, "wingspan-(cm)": 286.55723370140066, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11118.201610516982, "wingspan-(cm)": 289.86652005061836, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12018.63894948806, "wingspan-(cm)": 281.85202987389243, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12237.844621923621, "wingspan-(cm)": 301.2959328153797, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12013.548548628612, "wingspan-(cm)": 265.8584165574057, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 10364.470600619177, "wingspan-(cm)": 319.14893395150995, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 10955.790122290684, "wingspan-(cm)": 268.28845831520624, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12613.038881687546, "wingspan-(cm)": 291.9537268302905, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12736.205213323823, "wingspan-(cm)": 304.2404129699148, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 13026.92143939979, "wingspan-(cm)": 259.7721692431621, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 10567.809388941074, "wingspan-(cm)": 288.8068911959883, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 10158.811699813283, "wingspan-(cm)": 294.515741956821, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12366.093226167304, "wingspan-(cm)": 264.7265005747223, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11668.22286494719, "wingspan-(cm)": 293.33586214168173, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11310.782021910247, "wingspan-(cm)": 279.72617397129153, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 14034.607561504934, "wingspan-(cm)": 288.1069822442963, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11449.285588085408, "wingspan-(cm)": 319.85410474631135, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12750.453330326842, "wingspan-(cm)": 297.8449670678113, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 10693.00766091918, "wingspan-(cm)": 289.7548189586377, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12580.573335794274, "wingspan-(cm)": 283.76275496239026, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 10895.476907337707, "wingspan-(cm)": 269.622455948603, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12690.121470224707, "wingspan-(cm)": 282.2835516294682, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12686.890066138405, "wingspan-(cm)": 286.7590981999511, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 10433.31247042161, "wingspan-(cm)": 296.33570330632966, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12904.974121466681, "wingspan-(cm)": 273.58935603451647, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12778.822399323068, "wingspan-(cm)": 308.5536182778534, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12428.232870596741, "wingspan-(cm)": 286.5457298235934, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12108.871989897914, "wingspan-(cm)": 279.43372700397464, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12028.283634823074, "wingspan-(cm)": 281.1293731837224, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11421.174175209011, "wingspan-(cm)": 301.0549275352732, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 10800.54880080607, "wingspan-(cm)": 296.53800878772364, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 10294.04799426183, "wingspan-(cm)": 316.6399037826015, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12369.163957107006, "wingspan-(cm)": 297.69611568259467, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 13876.573426962166, "wingspan-(cm)": 307.5579047442221, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11623.096649831024, "wingspan-(cm)": 321.16568348375307, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 13831.936081825535, "wingspan-(cm)": 283.1611697117897, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12003.017434031213, "wingspan-(cm)": 299.73759390882026, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11923.976534275376, "wingspan-(cm)": 287.3782766832275, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 12003.9575939876, "wingspan-(cm)": 305.2589651487675, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 11814.985889102885, "wingspan-(cm)": 281.000254327267, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}, {"weight-(gm)": 9512.848464777231, "wingspan-(cm)": 313.6425008647881, "species": -1, "url": "https://raw.githubusercontent.com/pabloinsente/nn-mod-cog/master/notebooks/images/condor.png"}]}}, {"mode": "vega-lite"});
</script>



From **Chart 3** is clear that there is no way to trace a line to separate albatross from condors based on the available features. Let's train the ADALINE to test performance on this dataset.  


```python
df_shuffle2 = df2.sample(frac=1, random_state=1).reset_index(drop=True)
X = df_shuffle2[['weight-(gm)','wingspan-(cm)']].to_numpy()
y = df_shuffle2['species'].to_numpy()
```


```python
w_3, mse_3 = fit(X, y, eta=1e-10, n_iter=12)
```


```python
y_pred = predict(X,w_3)
num_correct_predictions = (y_pred == y).sum()
accuracy = (num_correct_predictions / y.shape[0]) * 100
print('Perceptron accuracy: %.2f%%' % accuracy)
```

    Perceptron accuracy: 50.00%



```python
mse_df["mse_3"] = mse_3
alt.Chart(mse_df).mark_line().encode(
    x="time-step", y="mse_3"
).properties(
    title='Chart 4'
)
```





<div id="altair-viz-a66ae9dcf47549c793a8e5afc4666a76"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    const outputDiv = document.getElementById("altair-viz-a66ae9dcf47549c793a8e5afc4666a76");
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.0.2?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function loadScript(lib) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = paths[lib];
        s.async = true;
        s.onload = () => resolve(paths[lib]);
        s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
        document.getElementsByTagName("head")[0].appendChild(s);
      });
    }
    
    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }
    
    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }
    
    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else if (typeof vegaEmbed === "function") {
      displayChart(vegaEmbed);
    } else {
      loadScript("vega")
        .then(() => loadScript("vega-lite"))
        .then(() => loadScript("vega-embed"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-360eeddc0a67da21bdf42987ae0eb5ca"}, "mark": "line", "encoding": {"x": {"type": "quantitative", "field": "time-step"}, "y": {"type": "quantitative", "field": "mse_3"}}, "title": "Chart 4", "$schema": "https://vega.github.io/schema/vega-lite/v4.0.2.json", "datasets": {"data-360eeddc0a67da21bdf42987ae0eb5ca": [{"mse": 1657.7059195090953, "time-step": 0, "mse-2": 1657.7059195090953, "mse_3": 4344.658702984796}, {"mse": 645.6347772640765, "time-step": 1, "mse-2": 384923.86858839454, "mse_3": 51831.8270984244}, {"mse": 251.58682683094383, "time-step": 2, "mse-2": 89398787.1664703, "mse_3": 618536.5051366336}, {"mse": 98.16500374231856, "time-step": 3, "mse-2": 20762937503.05355, "mse_3": 7381504.296606026}, {"mse": 38.43049519101968, "time-step": 4, "mse-2": 4822208321414.161, "mse_3": 88089737.52661656}, {"mse": 15.172960286729813, "time-step": 5, "mse-2": 1119961618729396.1, "mse_3": 1051249567.9013629}, {"mse": 6.1176641091291435, "time-step": 6, "mse-2": 2.6011195365761754e+17, "mse_3": 12545453092.351713}, {"mse": 2.591981953502288, "time-step": 7, "mse-2": 6.041120276277085e+19, "mse_3": 149715536909.12344}, {"mse": 1.2192442640539063, "time-step": 8, "mse-2": 1.4030548646174208e+22, "mse_3": 1786682539822.8384}, {"mse": 0.6847513186984904, "time-step": 9, "mse-2": 3.258605793460144e+24, "mse_3": 21321998798780.074}, {"mse": 0.47662733844914024, "time-step": 10, "mse-2": 7.568137201867329e+26, "mse_3": 254453503989904.72}, {"mse": 0.39557448463381123, "time-step": 11, "mse-2": 1.7577057286659103e+29, "mse_3": 3036609574166602.0}]}}, {"mode": "vega-lite"});
</script>



The ADALINE accuracy drops to 50%, basically random guessing, given the same $\eta$ and iterations, but different a dataset (a nonlinearly separable one). You would be able to obtain better accuracy by a combination of tweaking $\eta$ and iterations, but it would never reach zero. Yet, it illustrates the fact that now the problem became really hard for the network. Finally, **Chart 4** shows how baffling is the MSE over iterations with these parameters.

### ADALINE limitations summary

Summarizing, the ADALINE :
- is vulnerable to error/gradient explosion if an inappropriate learning rate has been chosen. 
- it can't overcome the linear separability problem. It is a linear model, still.

## Conclusions

With the ADALINE, Widrow and Hoff introduced for the first time the application of learning via gradient descent in the context of neural network models. If you are familiar with the contemporary literature in neural networks, you may be thinking I'm wrong or even lying. "Everybody knows that Rumelhart, Hinton, and Williams were the first ones on doing this in 1985". What Rumelhart, Hinton, and Williams introduced, was a generalization of the gradient descend method, the so-called "backpropagation" algorithm, in the context of training multi-layer neural networks with non-linear processing units. Hence, it wasn't actually the first gradient descent strategy ever applied, just the more general. 

We saw in practice how training a neural network with the Widrow and Hoff approach dramatically reduced the training time compared to the perceptron learning procedure. 

Although the ADALINE had a great impact in areas like telecommunications and engineering, its influence in the cognitive science community was very limited. Nonetheless, the methodological innovation introduced by Widrow and Hoff meant a step forward in what today we know is standard algorithms to train neural networks. 

## References

- Talbert, L. R., Groner, G. F., & Koford, J. S. (1963). Real-Time Adaptive Speech-Recognition System. The Journal of the Acoustical Society of America, 35(5), 807–807.


- Widrow, B., & Hoff, M. E. (1960). Adaptive switching circuits (No. TR-1553-1). Stanford Univ Ca Stanford Electronics Labs.

- Widrow, B., & Lehr, M. A. (1990). 30 years of adaptive neural networks: perceptron, madaline, and backpropagation. Proceedings of the IEEE, 78(9), 1415-1442.

- Widrow, B., & Lehr, M. A. (1995). Perceptrons, Adalines, and backpropagation. The handbook of brain theory and neural networks, 719-724.
