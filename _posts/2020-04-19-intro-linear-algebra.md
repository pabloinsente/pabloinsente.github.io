---
title: Introduction to Linear Algebra for Applied Machine Learning
published: true
mathjax: true
---

<<<***This is a work in progress as 2020-04-19***>>>

Linear algebra is to machine learning as flour to bakery: **every machine learning model is based in linear algebra, as every cake is based in flour**. It is not the only ingredient, of course. Machine learning models need vector calculus, probability, and optimization, as cakes need sugar, eggs, and butter. Applied machine learning, like bakery, is essentially about combining these mathematical ingredients in clever ways to create useful (tasty?) models. 

This document contains **introductory level linear algebra notes for applied machine learning**. It is meant as a reference rather than a comprehensive review. If you ever get confused by matrix multiplication, don't remember what was the $L^2$ norm, or the conditions for linear independence, this can serve as a quick reference. It also a good introduction for people that don't need a deep understanding of linear algebra, but still want to learn about the fundamentals to read about machine learning or to use pre-packaged machine learning solutions. Further, it is a good source for people that learned linear algebra a while ago and need a refresher.

These notes are based in a series of (mostly) freely available textbooks, video lectures, and classes I've read, watched and taken in the past. If you want to obtain a deeper understanding or to find exercises for each topic, you may want to consult those sources directly. 

**Free resources**:

- **Mathematics for Machine Learning** by Deisenroth, Faisal, and Ong. 1st Ed. [Book link](https://mml-book.github.io/).
- **Linear Algebra Ch. in Deep Learning** by Goodfellow, Bengio, and Courville. 1st Ed. [Chapter link](https://www.deeplearningbook.org/contents/linear_algebra.html).
- **Linear Algebra Ch. in Dive into Deep Learning** by Zhang, Lipton, Li, And Smola. [Chapter link](https://d2l.ai/chapter_preliminaries/linear-algebra.html).
- **Prof. Pavel Grinfeld's Linear Algebra Lectures** at Lemma. [Videos link](https://www.lem.ma/books/AIApowDnjlDDQrp-uOZVow/landing).
- **Prof. Gilbert Strang's Linear Algebra Lectures** at MIT. [Videos link](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/video-lectures/).
- **Salman Khan's Linear Algebra Lectures** at Khan Academy. [Videos link](https://www.khanacademy.org/math/linear-algebra).
- **3blue1brown's Linear Algebra Series** at YouTube. [Videos link](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab).

**Not-free resources**:

- **Introduction to Linear Algebra** by Gilbert Strang. 5th Ed. [Book link](https://www.amazon.com/Introduction-Linear-Algebra-Gilbert-Strang/dp/0980232775).
- **No Bullshit Guide to Linear Algebra** by Ivan Savov. 2nd Ed. [Book Link](https://www.amazon.com/No-bullshit-guide-linear-algebra/dp/0992001021).

I've consulted all these resources at one point or another. Pavel Grinfeld's lectures are my absolute favorites. Salman Khan's lectures are really good for absolute beginners (they are long though). The famous 3blue1brown series in linear algebra is delightful to watch and to get a solid high-level view of linear algebra.

The Linear Algebra Chapter in Goodfellow et al is a nice and concise introduction, but it may require some previous exposure to linear algebra concepts. Deisenroth et all book is probably the best and most comprehensive source for linear algebra for machine learning I've found, although it assumes that you are good at reading math (and at math more generally). Savov's book it's also great for beginners but requires time to digest. Professor Strang lectures are great too but I won't recommend it for absolute beginners.

I'll do my best to keep notation consistent. Nevertheless, learning to adjust to changing or inconsistent notation is a useful skill, since most authors will use their own preferred notation, and everyone seems to think that its/his/her own notation is better.

To make everything more dynamic and practical, I'll introduce bits of Python code to exemplify each mathematical operation (when possible) with `NumPy`, which is the facto standard package for scientific computing in Python.

Finally, keep in mind this is created by a non-mathematician for (mostly) non-mathematicians. If you find any mistake in notes feel free to reach me out at pcaceres@wisc.edu and to https://pablocaceres.org/ so I can correct the issue.

# Table of contents

**[Vectors](#vectors)**:
- [Types of vectors](#types-of-vectors)
    - [Geometric vectors](#geometric-vectors)
    - [Polynomials](#polynomials)
    - [Elements of $\mathbb{R}^n$](#elements-of-$\mathbb{R}$)
- [Vector dimensions and coordinate system](#vector-dimensions-and-coordinates-systems)
- [Basic vector operations](#basic-vector-operations)
    - [Vector-vector addition](#vector-vector-addition)
    - [Vector-scalar multiplication: scaling](#vector-scalar-multiplication:-scaling)
    - [Linear combinations of vectors](#linear-combinations-of-vectors)
    - [Vector-vector multiplication: dot product](#vector-vector-multiplication:-dot-product)
- [Vector spaces](#vector-spaces)
    - [Vector space](#vector-space)
    - [Vector span](#vector-span)
    - [Vector subspaces](#vector-subspaces)
- [Linear dependence and independence](#linear-dependence-and-independence)
- [Vector norms](#vector-norms)
    - [Euclidean norm: $L_2$](#euclidean-norm:-$L_2$)
    - [Manhattan norm: $L_1$](#manhattan-norm-$L_1$)
    - [Max norm: $L_\infty$](#max-norm-$L_\infty$)
- [Vector inner product, length, and distance](#vector-inner-product-length-and-distance)
- [Vector angles and orthogonality](#vector-angles-and-orthogonality)
- [Systems of linear equations](#systems-of-linear-equations)

**[Matrices](#matrices)**:

- [Basic matrix operations](#basic-matrix-operations)
    - [Matrix-matrix addition](#matrix-matrix-addition)
    - [Matrix-scalar multiplication](#matrix-scalar-multiplication)
    - [Matrix-vector multiplication: dot product](#matrix-vector-multiplication-dot-product)
    - [Matrix-matrix multiplication](#matrix-matrix-multiplication)
    - [Hadamard product](#hadamard-product)
- [Matrices as systems of linear equations](#matrices-as-systems-of-linear-equations)

**Future sections**:

- Solving systems of linear equations with Gaussian elimination
- Matrix identity
- Matrix inverse
- Matrix transpose
- Matrix basis and rank
- Linear mappings
- Affine spaces
- Affine transformations
- Orthogonal projections
- Matrices norms
- Eigenvalues and eigenvectors
- Eigendecomposition and diagonalization
- Singular value decomposition
- Eigenvalue decomposition vs Singular value decomposition
- LU decomposition


```python
# Libraries for this section 
import numpy as np
import pandas as pd
import altair as alt
```

# Vectors

Linear algebra is the study of vectors. Vectors are the most fundamental mathematical object in machine learning. We use them to **represent attributes of entities**: age, sex, test scores, etc. We represent vectors by a bold lower-case letter like $\bf{v}$ or as a lower-case letter with an arrow on top like $\vec{v}$.

Vectors are a type of mathematical object that can be **added together** and/or **multiplied by a number** to obtain another object of **the same kind**. For instance, if we have a vector $\bf{x} = \text{age}$ and a second vector $\bf{y} = \text{age}$, we can add them together and obtain a third vector $\bf{z} = x + y$ of *the same kind*. We can also multiply $2 \times \bf{x}$ to obtain $2\bf{x}$, again, a vector of *the same kind*. 

## Types of vectors

Vectors come in three flavors: (1) **geometric vectors**, (2) **polynomials**, (3) and **elements of $\mathbb{R^n}$ space**. We will defined each one next.

### Geometric vectors

**Geometric vectors are oriented segments**. Therse are the kind of vectors you probably learned about in high-school physics and geometry. Many linear algebra concepts come from the geometric point of view of vectors: space, plane, distance, etc.

**Fig. 1: Geometric vectors**


<img src="/assets/post-10/b-geometric-vectors.svg">


### Polynomials

**A polynomial is an expression like $f(x) = x^2 + y + 1$**. This is, a expression adding multiple "terms" (nomials). Polynomials are vectors because they meet the definition of a vector: they can be added together to get another polynomial, and they can be multiplied together to get another polynomial. 

$$
\text{function addition is valid} \\
f(x) + g(x)\\
$$
$$
and\\
$$
$$
\text{multiplying by a scalar is valid} \\
5 \times f(x)
$$

**Fig. 2: Polynomials**


<img src="/assets/post-10/b-polynomials-vectors.svg">


### Elements of $\mathbb{R}$

**Elements of $\mathbb{R^n}$ are sets of real numbers**. This type of representation is arguably the most important for applied machine learning. It is how data is commonly represented in computers to build machine learning models. For instance, a vector in $\mathbb{R^3}$ takes the shape of:

$$\bf{x}=
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}
\in \mathbb{R^3}
$$

Indicating that it contains three dimensions.

$$
\text{addition is valid} \\
\phantom{space}\\
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix} +
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}=
\begin{bmatrix}
2 \\
4 \\
6
\end{bmatrix}\\
$$
$$
and\\
$$
$$
\text{multiplying by a scalar is valid} \\
\phantom{space}\\
5 \times
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix} = 
\begin{bmatrix}
5 \\
10 \\
15
\end{bmatrix}
$$

In `NumPy` vectors are represented as n-dimensional arrays. To create a vector in $\mathbb{R^3}$:


```python
x = np.array([[1],
              [2],
              [3]])
```

We can inspect the vector shape by:


```python
x.shape # (3 dimensions, 1 element on each)
```




    (3, 1)




```python
print(f'A 3-dimensional vector:\n{x}')
```

    A 3-dimensional vector:
    [[1]
     [2]
     [3]]


## Vector dimensions and coordinate system

Vectors can have any number of dimensions. The most common are the 2-dimensional cartesian plane, and the 3-dimensional space. Vectors in 2 and 3 dimensions are used often for pedgagogical purposes since we can visualize them as geometric vectors. Nevetheless, most problems in machine learning entail more dimensions, sometiome hundreds or thousands of dimensions. The notation for a vector $\bf{x}$ of arbitrary dimensions, $n$ is:

$$
\bf{x} = 
\begin{bmatrix}
x_1 \\ x_2 \\ \vdots \\ x_n
\end{bmatrix}
\in \mathbb{R}^n
$$

Vectors dimensions map into **coordinate systems or perpendicular axes**. Coordinate systems have an origin at $(0,0,0)$, hence, when we define a vector:

$$\bf{x} = \begin{bmatrix} 3 \\ 2 \\ 1 \end{bmatrix} \in \mathbb{R}^3$$

we are saying: starting from the origin, move 3 units in the 1st perpendicular axis, 2 units in the 2nd perpendicular axis, and 1 unit in the 3rd perpendicular axis. We will see later that when we have a set of perpendicular axes we obtain the basis of a vector space.

**Fig. 3: Coordinate systems**


<img src="/assets/post-10/b-coordinate-system.svg">


## Basic vector operations

### Vector-vector addition 

We used vector-vector addition to define vectors without defining vector-vector addition. Vector-vector addition is an element-wise operation, only defined for vectors of the same size (i.e., number of elements). Consider two vectors of the same size, then: 

$$
\bf{x} + \bf{y} = 
\begin{bmatrix}
x_1\\
\vdots\\
x_n
\end{bmatrix}+
\begin{bmatrix}
y_1\\
\vdots\\
y_n
\end{bmatrix} =
\begin{bmatrix}
x_1 + y_1\\
\vdots\\
x_n + y_n
\end{bmatrix}
$$

For instance:

$$
\bf{x} + \bf{y} = 
\begin{bmatrix}
1\\
2\\
3
\end{bmatrix}+
\begin{bmatrix}
1\\
2\\
3
\end{bmatrix} =
\begin{bmatrix}
1 + 1\\
2 + 2\\
3 + 3
\end{bmatrix} =
\begin{bmatrix}
2\\
4\\
6
\end{bmatrix}
$$

In `NumPy`, we add two vectors of the same with the `+` operator or the `add` method:


```python
x = y = np.array([[1],
                  [2],
                  [3]])
```


```python
x + y
```




    array([[2],
           [4],
           [6]])




```python
np.add(x,y)
```




    array([[2],
           [4],
           [6]])



### Vector-scalar multiplication: scaling

Vector-scalar multiplication is an element-wise operation. It's defined as:

$$
\alpha \bf{x} = 
\begin{bmatrix}
\alpha \bf{x_1}\\
\vdots \\
\alpha \bf{x_n}
\end{bmatrix}
$$

Consider $\alpha = 2$ and $\bf{x} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$:

$$
\alpha \bf{x} = 
\begin{bmatrix}
2 \times 1\\
2 \times 2\\
2 \times 3
\end{bmatrix} = 
\begin{bmatrix}
2\\
4\\
6
\end{bmatrix}
$$

In `NumPy`, we compute scalar-vector multiplication with the `*` operator:


```python
alpha = 2
x = np.array([[1],
             [2],
             [3]])
```


```python
alpha * x
```




    array([[2],
           [4],
           [6]])



### Linear combinations of vectors

There are only two legal operations with vectors in linear algebra: **addition** and **multiplication by numbers**. When we combine those, we get a **linear combination**.

$$
\alpha \bf{x} + \beta \bf{y} = 
\alpha
\begin{bmatrix}
x_1 \\ 
x_2
\end{bmatrix}+
\beta
\begin{bmatrix}
y_1 \\ 
y_2
\end{bmatrix}=
\begin{bmatrix}
\alpha x_1 + \alpha x_2\\ 
\beta y_1 + \beta y_2
\end{bmatrix}
$$

Consider $\alpha = 2$, $\beta = 3$, $\bf{x}=\begin{bmatrix}2 \\ 3\end{bmatrix}$, and $\begin{bmatrix}4 \\ 5\end{bmatrix}$.

We obtain:

$$
\alpha \bf{x} + \beta \bf{y} = 
2
\begin{bmatrix}
2 \\ 
3
\end{bmatrix}+
3
\begin{bmatrix}
4 \\ 
5
\end{bmatrix}=
\begin{bmatrix}
2 \times 2 + 2 \times 4\\ 
2 \times 3 + 3 \times 5
\end{bmatrix}=
\begin{bmatrix}
10 \\
21
\end{bmatrix}
$$

Another way to express linear combinations you'll see often is with summation notation. Consider a set of vectors $x_1, ..., x_k$ and scalars $\beta_1, ..., \beta_k \in \mathbb{R}$, then:   

$$
\sum_{i=1}^k \beta_i x_i := \beta_1x_1 + ... + \beta_kx_k
$$

Note that $:=$ means "*is defined as*".

Linear combinations are the most fundamental operation in linear algebra. Everything in linear algebra results from linear combinations. For instance, linear regression is a linear combination of vectors. **Fig. 1** shows an example of how adding two geometrical vectors looks like for intuition.

In `NumPy`, we do linear combinations as:


```python
a, b = 2, 3
x , y = np.array([[2],[3]]), np.array([[4], [5]])
```


```python
a*x + b*y
```




    array([[16],
           [21]])



### Vector-vector multiplication: dot products

We covered vector addition and multiplication by scalars. Now I will define vector-vector multiplication, commonly known as a **dot product** or **inner product**. The dot product of $\bf{x}$ and $\bf{y}$ is defined as: 

$$
\bf{x} \cdot \bf{y} :=
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
\begin{bmatrix}
y_1 \\
y_2
\end{bmatrix} =
x_1 \times y_1 + x_2 \times y_2 
$$

For instance:

$$
\bf{x} \cdot \bf{y} =
\begin{bmatrix}
-2 \\
2
\end{bmatrix}
\begin{bmatrix}
4 \\
-3
\end{bmatrix} =
-2 \times 4 + 2 \times -3 = (-8) + (-6) = -14  
$$

Dot products are so important in machine learning, that after a while they become second nature for practitioners.

To multiply two vectors with dimensions (rows=2, cols=1) in `Numpy`, we need to transpose the first vector at using the `@` operator:


```python
x, y = np.array([[-2],[2]]), np.array([[4],[-3]])
```


```python
x.T @ y
```




    array([[-14]])



## Vector spaces

### Vector space

In its more general form, a **vector space**, also known as **linear space**, is a collection of objects that follow the rules defined for vectors in $\mathbb{R}^n$. We mentioned those rules when we defined vectors: they can be added together and multiplied by scalars, and return vectors of the same type. More colloquially, a vector space is the set of proper vectors and all possible linear combinatios of the vector set. In addition, vector addition and multiplication must follow these eight rules: 

1. commutativity: $x + y = y + x$
2. associativity: $x + (y + x) = (y + x) + z$
3. unique zero vector such that: $x + 0 = x$ $\forall$ $x$ 
4. $\forall$ $x$ there is a unique vector $x$ such that $x + -x = 0$
5. identity element of scalar multiplication: $1x = x$
6. distributivity of scalar multiplication w.r.t vector addition: $x(y + z) = xz + zy$
7. $x(yz) = (xy)z$
8. $(y + z)x = yx + zx$

In my experience remembering  these properties is not really important, but it's good to know that such rules exist.

### Vector span

Consider the vectors $\bf{x}$ and $\bf{y}$ and the scalars $\alpha$ and $\beta$. If we take *all* possible linear combinations of $\alpha \bf{x} + \beta \bf{y}$ we would obtain the **span** of such vectors. This is easier to grasp when you think about geometric vectors. If our vectors $\bf{x}$ and $\bf{y}$ point into **different directions** in the 2-dimensional space, we get that the $span(x,y)$ is equal to **the entire 2-dimensional plane**, as shown in the middle-pane in **Fig. 4**. Just imagine having an unlimited number of two types of sticks: one pointing vertically, and one pointing horizontally. Now, you can reach any point in the 2-dimensional space by simply combining the necessary number of vertical and horizontal sticks (including taking fractions of sticks). 

**Fig. 4: Vector Span**


<img src="/assets/post-10/b-vector-span.svg">


What would happen if the vectors point in the same direction? Now, if you combine them, you just can **span a line**, as shown in the left-pane in **Fig. 4**. If you have ever heard of the term "multicollinearity", it's closely related to this issue: when two variables are "colinear" they are pointing in the same direction, hence they provide redundant information, so can drop one without information loss.

With three vectors pointing into different directions, we can span the entire 3-dimensional space or a **hyper-plane**, as in the right-pane of **Fig. 4**. Note that the sphere is just meant as a 3-D reference, not as a limit.

Four vectors pointing into different directions will span the 4-dimensional space, and so on. From here our geometrical intuition can't help us. This is an example of how linear algebra can describe the behavior of vectors beyond our basics intuitions. 

### Vector subspaces

A **vector subspace (or linear subspace) is a vector space that lies within a larger vector space**. These are also known as linear subspaces. For a vector to be a valid subspace it has to meet **three conditions**:

1. Contains the zero vector
2. Closure under multiplication 
3. Closure under addition

**Fig. 5: Vector subspaces**


<img src="/assets/post-10/b-vector-subspace.svg">


Consider the following questions: Is $\bf{x}=\begin{bmatrix} 1 \\ 1 \end{bmatrix}$ a valid subspace of $\mathbb{R^2}$? Let's evaluate $\bf{x}$ on the three conditions:

**Contains the zero vector**: it does. Remember that the span of a vector are all linear combinations of such a vector. Therefore, we can simply multiply by $0$ to get $\begin{bmatrix}0 \\ 0 \end{bmatrix}$:

$$
\bf{x}\times 0=0
\begin{bmatrix}
1 \\ 
1 
\end{bmatrix}
=
\begin{bmatrix}
0 \\ 
0 
\end{bmatrix}
$$

**Closure under multiplication** implies that if take any vector belonging to $\bf{x}$ and multiply by any real scalar $\alpha$, the resulting vector stays within the span of $\bf{x}$. Algebraically is easy to see that we can multiply $\begin{bmatrix} 1 \\ 1 \end{bmatrix}$ by any scalar $\alpha$, and the resulting vector remains in the 2-dimensional plane (i.e., the span of $\bf{x}$).

**Closure under addition** implies that if we add together any vectors belonging to $\bf{x}$, the resulting vector remains within the span of $\bf{x}$. Again, algebraically is clear that if we add any vectors from $\bf{x}$ span, the resulting vector will remain in such a span. There is no way to get to $\mathbb{R^3}$ or $\mathbb{R^4}$ or any space outside the two-dimensional plane.

## Linear dependence and independence 

The left-pane shows a triplet of **linearly dependent** vectors, whereas the right-pane shows a triplet of **linearly independent** vectors.

**Fig. 6: Linear dependence and independence**


<img src="/assets/post-10/b-linear-independence.svg">


A set of vectors is **linearly dependent** if at least one vector can be obtained as a linear combination of other vectors in the set. As you can see in the left pane, we can combine vectors $x$ and $y$ to obtain $z$. 

There is more rigurous (but slightly harder to grasp) definition of linear dependence. Consider a set of vectors $x_1, ..., x_k$ and scalars $\beta \in \mathbb{R}$. If there is a way to get $0 = \sum_{i=1}^k \beta_i x_i$ with at least one $\beta \ne 0$, we have linearly dependent vectors.

A set of vectors is **linearly independent** if none vector can be obtained a linear combination of other vectors in the set. As you can see in the right pane, there is no way for us to combine vectors $x$ and $y$ to obtain $z$. 

Again, consider a set of vectors $x_1, ..., x_k$ and scalars $\beta \in \mathbb{R}$. If the only way to get $0 = \sum_{i=1}^k \beta_i x_i$ requires all $\beta_1, ..., \beta_k = 0$, the we have linearly independent vectors. 

The importance of the concepts of linear dependence and independence will become clearer in more advanced topics. For now, the important points to remember are: linearly dependent vectors contain **redundant information**, whereas linearly independent vectors do not.

## Vector norms

Measuring vectors is another important operation in machine learning applications. Intuitively, we can think about the **norm** or the **length** of a vector as the distance between its "origin" and its "end".  

Norms "map" vectors to non-negative values. In this sense are functions that assign length $\lVert \bf{x} \rVert \in \mathbb{R^n}$ to a vector $\bf{x}$. To be valid, a norm has to satisfy these properties (keep in mind these properties are a bit abstruse to understand):

1. **Absolutely homogeneous**: $\forall \alpha \in \mathbb{R},  \lVert \alpha \bf{x} \rVert = \vert \alpha \Vert \lVert \bf{x} \rVert$. In words: for all real-valued scalars, the norm scales proportionally with the value of the scalar.
2. **Triangle inequality**: $\lVert \bf{x} + \bf{y} \rVert \le \lVert \bf{x} \rVert + \lVert \bf{x} \rVert $. In words: in geometric terms, for any triangle the sum of any two sides must be greater or equal to the lenght of the third side. This is easy to see experimentally: grab a piece of rope, form triangles of different sizes, measure all the sides, and test this property.
3. **Positive definite**: $\lVert \bf{x} \rVert \ge 0$ and $ \lVert \bf{x} \rVert = 0 \Longleftrightarrow \bf{x}= 0$. In words: the length of any $\bf{x}$ has to be a positive value (i.e., a vector can't have negative length), and a length of $0$ occurs only of $\bf{x}=0$ 

Grasping the meaning of these three properties may be difficult at this point, but they probably become clearer as you improve your understanding of linear algebra.

**Fig. 7: Vector norms**


<img src="/assets/post-10/b-l2-norm.svg">


### Euclidean norm: $L_2$

The Euclidean norm is one of the most popular norms in machine learning. It is so widely used that sometimes is refered simply as "the norm" of a vector. Is defined as:

$$
\lVert \bf{x} \rVert_2 := \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{x^Tx} 
$$

Hence, in **two dimensions** the $L^2$ norm is:

$$
\lVert \bf{x} \rVert_2 \in \mathbb{R^2} = \sqrt {x_1^2  \cdot x_2^2 } 
$$

Which is equivalent to the formula for the hypotenuse a triangle with sides $x_1^2$ and $x_2^2$. 

The same pattern follows for higher dimensions of $\mathbb{R^n}$

In `NumPy`, we can compute the $L^2$ norm as:


```python
x = np.array([[3],[4]])

np.linalg.norm(x, 2)
```




    5.0



If you remember the first "Pythagorean triple", you can confirm that the norm is correct.

### Manhattan norm: $L_1$

The Manhattan or $L_1$ norm gets its name in analogy to measuring distances while moving in Manhattan, NYC. Since Manhattan has a grid-shape, the distance between any two points is measured by moving in vertical and horizontals lines (instead of diagonals as in the Euclidean norm). It is defined as:

$$
\lVert \bf{x} \rVert_1 := \sum_{i=1}^n \vert x_i \vert 
$$

Where $\vert x_i \vert$ is the absolute value. The $L_1$ norm is preferred when discriminating between elements that are exactly zero and elements that are small but not zero.   

In `NumPy` we compute the $L_1$ norm as


```python
x = np.array([[3],[-4]])

np.linalg.norm(x, 1)
```




    7.0



Is easy to confirm that the sum of the absolute values of $3$ and $-4$ is $7$.

### Max norm: $L_\infty$

The max norm or infinity norm is simply the absolute value of the largest element in the vector. It is defined as:

$$
\lVert \bf{x} \rVert_\infty := max_i \vert x_i \vert 
$$

Where $\vert x_i \vert$  is the absolute value. For instance, for a vector with elements $\bf{x} = \begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$, the $\lVert \bf{x} \rVert_\infty = 3$

In `NumPy` we compute the $L_\infty$ norm as:


```python
x = np.array([[3],[-4]])

np.linalg.norm(x, np.inf)
```




    4.0



## Vector inner product, length, and distance.

For practical purposes, inner product and length are used as equivalent to dot product and norm, although technically are not the same. 

**Inner products** are a more general concept that dot products, with a series of additional properties (see [here](https://en.wikipedia.org/wiki/Inner_product_space#Elementary_properties)). In other words, every dot product is an inner product, but not every inner product is a dot product. The notation for the inner product is usually a pair of angle brackets as $\langle  .,. \rangle$ as. For instance, the scalar inner product is defined as:

$$
\langle x,y \rangle := x\cdot y
$$

In $\mathbb{R}^n$ the inner product is a dot product defined as:

$$
\Bigg \langle
\begin{bmatrix} 
x_1 \\ 
\vdots\\
x_n
\end{bmatrix},
\begin{bmatrix} 
y_1 \\ 
\vdots\\
y_n
\end{bmatrix}
\Bigg \rangle :=
x \cdot y = \sum_{i=1}^n x_iy_i
$$

**Length** is a concept from geometry. We say that geometric vectors have length and that vectors in $\mathbb{R}^n$ have norm. In practice, many machine learning textbooks use these concepts interchangeably. I've found authors saying things like "we use the $l_2$ norm to compute the *length* of a vector". For instance, we can compute the length of a directed segment (i.e., geometrical vector) $\bf{x}$ by taking the square root of the inner product with itself as:

$$
\lVert x \rVert = \sqrt{\langle x,x \rangle} = \sqrt{x\cdot y} = x^2 + y^2  
$$

**Distance** is a relational concept. It refers to the length (or norm) of the difference between two vectors. Hence, we use norms and lengths to measure the distance between vectors. Consider the vectors $\bf{x}$ and $\bf{y}$, we define the distance $d(x,y)$ as:

$$
d(x,y) := \lVert x - y \rVert = \sqrt{\langle x - y, x - y \rangle}
$$

When the inner product $\langle x - y, x - y \rangle$ is the dot product, the distance equals to the Euclidean distance.

In machine learning, unless made explicit, we can safely assume that an inner product refers to the dot product. We already reviewed how to compute the dot product in `NumPy`:


```python
x, y = np.array([[-2],[2]]), np.array([[4],[-3]])
x.T @ y 
```




    array([[-14]])



As with the inner product, usually, we can safely assume that **distance** stands for the Euclidean distance or $L_2$ norm unless otherwise noted. To compute the $L_2$ distance between a pair of vectors:


```python
distance = np.linalg.norm(x-y, 2)
print(f'L_2 distance : {distance}')
```

    L_2 distance : 7.810249675906656


## Vector angles and orthogonality

The concepts of angle and orthogonality are also related to geometrical vectors. We saw that inner products allow for the definition of length and distance. In the same manner, inner products are used to define **angles** and **orthogonality**. 

In machine learning, the **angle** between a pair of vectors is used as a **measure of vector similarity**. To understand angles let's first look at the **Cauchy–Schwarz inequality**. Consider a pair of non-zero vectors $\bf{x}$ and $\bf{y}$ $\in \mathbb{R}^n$. The Cauchy–Schwarz inequality states that:

$$
\vert \langle x, y \rangle \vert \leq \Vert x \Vert \Vert y \Vert
$$

In words: *the absolute value of the inner product of a pair of vectors is less than or equal to the products of their length*. The only case where both sides of the expression are *equal* is when vectors are colinear, for instance, when $\bf{x}$ is a scaled version of $\bf{y}$. In the 2-dimensional case, such vectors would lie along the same line. 

The definition of the angle between vectors can be thought as a generalization of the **law of cosines** in trigonometry, which defines for a triangle with sides $a$, $b$, and $c$, and an angle $\theta$ are related as:

$$
c^2 = a^2 + b^2 - 2ab \cos \theta
$$

**Fig. 8: Law of cosines and Angle between vectors**


<img src="/assets/post-10/b-vector-angle.svg">


We can replace this expression with vectors lengths as: 

$$
\Vert x - y \Vert^2 = \Vert x \Vert^2 + \Vert y \Vert^2 - 2(\Vert x \Vert \Vert y \Vert) \cos \theta
$$

With a bit of algebraic manipulation, we can clear the previous equation to:

$$
\cos \theta = \frac{\langle x, y \rangle}{\Vert x \Vert \Vert y \Vert} 
$$

And there we have a **definition for (cos) angle $\theta$**. Further, from the Cauchy–Schwarz inequality we know that $\cos \theta$ must be:

$$
-1 \leq \frac{\langle x, y \rangle}{\Vert x \Vert \Vert y \Vert} \leq 1  
$$

This is a necessary conclusion (range between $\{-1, 1\}$) since the numerator in the equation always is going to be smaller or equal to the denominator.

In `NumPy`, we can compute the $\cos \theta$ between a pair of vectors as: 


```python
x, y = np.array([[1], [2]]), np.array([[5], [7]])

# here we translate the cos(theta) definition
cos_theta = (x.T @ y) / (np.linalg.norm(x,2) * np.linalg.norm(y,2))
print(f'cos of the angle = {np.round(cos_theta, 3)}')
```

    cos of the angle = [[0.988]]


We get that $\cos \theta \approx 0.988$. Finally, to know the exact value of $\theta$ we need to take the trigonometric inverse of the cosine function as:


```python
cos_inverse = np.arccos(cos_theta)
print(f'angle in radiants = {np.round(cos_inverse, 3)}')
```

    angle in radiants = [[0.157]]


We obtain $\theta \approx 0.157 $. To fo from radiants to degrees we can use the following formula:


```python
degrees = cos_inverse * ((180)/np.pi)
print(f'angle in degrees = {np.round(degrees, 3)}')
```

    angle in degrees = [[8.973]]


We obtain $\theta \approx 8.973^{\circ}$

**Orthogonality** is often used interchangeably with "independence" although they are mathematically different concepts. Orthogonality can be seen as a generalization of perpendicularity to vectors in any number of dimensions.

We say that a pair of vectors $\bf{x}$ and $\bf{y}$ are **orthogonal** if their inner product is zero, $\langle x,y \rangle = 0$. The notation for a pair of orthogonal vectors is $\bf{x} \perp \bf{y}$. In the 2-dimensional plane, this equals to a pair of vectors forming a $90^{\circ}$ angle.

Here is an example of orthogonal vectors

**Fig. 9: Orthogonal vectors**


<img src="/assets/post-10/b-orthogonal-vectors.svg">



```python
x = np.array([[2], [0]])
y = np.array([[0], [2]])

cos_theta = (x.T @ y) / (np.linalg.norm(x,2) * np.linalg.norm(y,2))
print(f'cos of the angle = {np.round(cos_theta, 3)}')
```

    cos of the angle = [[0.]]


We see that this vectors are **orthogonal** as $\cos \theta=0$. This is equal to  $\approx 1.57$ radiants and $\theta = 90^{\circ}$


```python
cos_inverse = np.arccos(cos_theta)
degrees = cos_inverse * ((180)/np.pi)
print(f'angle in radiants = {np.round(cos_inverse, 3)}\nangle in degrees ={np.round(degrees, 3)} ')
```

    angle in radiants = [[1.571]]
    angle in degrees =[[90.]] 


## Systems of linear equations

The purpose of linear algebra as a tool is to **solve systems of linear equations**. Informally, this means to figure out the right combination of linear segments to obtain an outcome. Even more informally, think about making pancakes: In what proportion ($w_i \in \mathbb{R}$) we have to mix ingredients to make pancakes? You can express this as a linear equation: 

$$
f_\text{flour} \times w_1 + b_\text{baking powder}  \times w_2 + e_\text{eggs}  \times w_3 + m_\text{milk} \times w_4 = P_\text{pancakes}
$$

The above expression describe *a* linear equation. A *system* of linear equations involve multiple equations that have to be solved **simultaneously**. Consider:

$$
x + 2y = 8 \\
5x - 3y = 1
$$

Now we have a system with two unknowns, $x$ and $y$. We'll see general methods to solve systems of linear equations later. For now, I'll give you the answer: $x=2$ and $y=3$. Geometrically, we can see that both equations produce a straight line in the 2-dimensional plane. The point on where both lines encounter is the solution to the linear system.


```python
df = pd.DataFrame({"x1": [0, 2], "y1":[8, 3], "x2": [0.5, 2], "y2": [0, 3]})
```


```python
equation1 = alt.Chart(df).mark_line().encode(x="x1", y="y1")
equation2 = alt.Chart(df).mark_line(color="red").encode(x="x2", y="y2")
equation1 + equation2
```





<div id="altair-viz-c53535f19fb746f5abec52b2bb7ee5d7"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-c53535f19fb746f5abec52b2bb7ee5d7") {
      outputDiv = document.getElementById("altair-viz-c53535f19fb746f5abec52b2bb7ee5d7");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.8.1?noext",
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "layer": [{"mark": "line", "encoding": {"x": {"type": "quantitative", "field": "x1"}, "y": {"type": "quantitative", "field": "y1"}}}, {"mark": {"type": "line", "color": "red"}, "encoding": {"x": {"type": "quantitative", "field": "x2"}, "y": {"type": "quantitative", "field": "y2"}}}], "data": {"name": "data-57ffab6a26a928c2ff17e40b29b8a272"}, "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json", "datasets": {"data-57ffab6a26a928c2ff17e40b29b8a272": [{"x1": 0, "y1": 8, "x2": 0.5, "y2": 0}, {"x1": 2, "y1": 3, "x2": 2.0, "y2": 3}]}}, {"mode": "vega-lite"});
</script>



# Matrices

Matrices are as fundamental as vectors in machine learning. With vectors, we can represent single variables as sets of numbers or instances. With matrices, we can represent sets of variables. In this sense, a matrix is simply an ordered **collection of vectors**. Conventionally, column vectors, but it's always wise to pay attention to the authors' notation when reading matrices. Since computer screens operate in two dimensions, matrices are the way in which we interact with data in practice.

More formally, we represent a matrix with a italicized upper-case letter like $\textit{A}$. In two dimensions, we say the matrix $\textit{A}$ has $m$ rows and $n$ columns. Each entry of $\textit{A}$ is defined as $a_{ij}$, $i=1,..., m,$ and $j=1,...,n$. A matrix $\textit{A} \in \mathbb{R^{m\times n}}$ is defines as:

$$
A :=
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n}\\
a_{21} & a_{22} & \cdots & a_{2n}\\
\vdots & \vdots & \ddots & \vdots\\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix},
a_{ij} \in \mathbb{R}
$$

In `Numpy`, we construct matrices with the `array` method: 


```python
A = np.array([[0,2],  # 1st row
              [1,4]]) # 2nd row

print(f'a 2x2 Matrix:\n{A}')
```

    a 2x2 Matrix:
    [[0 2]
     [1 4]]


## Basic Matrix operations

### Matrix-matrix addition 

We add matrices in a element-wise fashion. The sum of $\textit{A} \in \mathbb{R}^{m\times n}$ and $\textit{B} \in \mathbb{R}^{m\times n}$ is defined as:

$$
\textit{A} + \textit{B} := 
\begin{bmatrix}
a_{11} + b_{11} & \cdots & a_{1n} + b_{1n} \\
\vdots & \ddots & \vdots \\
a_{m1} + b_{m1} & \cdots & a_{mn} + b_{mn}
\end{bmatrix}
\in \mathbb{R^{m\times n}}
$$

For instance: 
$$
\textit{A} = 
\begin{bmatrix}
0 & 2 \\
1 & 4
\end{bmatrix} + 
\textit{B} = 
\begin{bmatrix}
3 & 1 \\
-3 & 2
\end{bmatrix}=
\begin{bmatrix}
0+3 & 2+1 \\
3+(-3) & 2+2
\end{bmatrix}=
\begin{bmatrix}
3 & 3 \\
-2 & 6
\end{bmatrix}
$$ 

In `Numpy`, we add matrices with the `+` operator or `add` method:


```python
A = np.array([[0,2],
              [1,4]])
B = np.array([[3,1],
              [-3,2]])
```


```python
A + B
```




    array([[ 3,  3],
           [-2,  6]])




```python
np.add(A, B)
```




    array([[ 3,  3],
           [-2,  6]])



### Matrix-scalar multiplication

Matrix-scalar multiplication is an element-wise operation. Each element of the matrix $\textit{A}$ is multiplied by the scalar $\alpha$. Is defined as:

$$
a_{ij} \times \alpha, \text{such that } (\alpha \textit{A})_{ij} = \alpha(\textit{A})_{ij}
$$

Consider $\alpha=2$ and $\textit{A}=\begin{bmatrix}1 & 2 \\3 & 4\end{bmatrix}$, then:

$$
\alpha \textit{A} =
2
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}=
\begin{bmatrix}
2\times 1 & 2\times 2 \\
2\times 3 & 2 \times4 
\end{bmatrix}=
\begin{bmatrix}
2 & 4 \\
6 & 8 
\end{bmatrix}
$$

In `NumPy`, we compute matrix-scalar multiplication with the `*` operator or `multiply` method:


```python
alpha = 2
A = np.array([[1,2],
              [3,4]])
```


```python
alpha * A
```




    array([[2, 4],
           [6, 8]])




```python
np.multiply(alpha, A)
```




    array([[2, 4],
           [6, 8]])



### Matrix-vector multiplication: dot product

Matrix-vector multiplication equals to taking the dot product of each column $n$ of a $\textit{A}$ with each element $\bf{x}$ resulting in a vector $\bf{y}$. Is defined as: 

$$
\textit{A}\cdot\bf{x}:=
\begin{bmatrix}
a_{11} & \cdots & a_{1n}\\
\vdots & \ddots & \vdots\\
a_{m1} & \cdots & a_{mn}
\end{bmatrix}
\begin{bmatrix}
x_1\\
\vdots\\
x_n
\end{bmatrix}=
x_1
\begin{bmatrix}
a_{11}\\
\vdots\\
a_{m1}
\end{bmatrix}+
x_2
\begin{bmatrix}
a_{12}\\
\vdots\\
a_{m2}
\end{bmatrix}+
x_n
\begin{bmatrix}
a_{1n}\\
\vdots\\
a_{mn}
\end{bmatrix}=
\begin{bmatrix}
y_1\\
\vdots\\
y_{mn}
\end{bmatrix}
$$

For instance:

$$
\textit{A}\cdot\bf{x}=
\begin{bmatrix}
0 & 2\\
1 & 4
\end{bmatrix}
\begin{bmatrix}
1\\
2
\end{bmatrix}=
1
\begin{bmatrix}
0 \\
1
\end{bmatrix}+
2
\begin{bmatrix}
2 \\
4
\end{bmatrix}=
\begin{bmatrix}
1\times0 + 2\times2 \\
1\times1 + 2\times4
\end{bmatrix}=
\begin{bmatrix}
4 \\
9
\end{bmatrix}
$$

In numpy, we compute the matrix-vector product with the `@` operator or `dot` method:


```python
A = np.array([[0,2],
              [1,4]])
x = np.array([[1],
              [2]])
```


```python
A @ x
```




    array([[4],
           [9]])




```python
np.dot(A, x)
```




    array([[4],
           [9]])



### Matrix-matrix multiplication

Matrix-matrix multiplication is a dot produt as well. To work, the number of columns in the first matrix $\textit{A}$ has to be equal to the number of rows in the second matrix $\textit{B}$. Hence, $\textit{A} \in \mathbb{R^{m\times n}}$ times $\textit{B} \in \mathbb{R^{n\times p}}$ to be valid. One way to see matrix-matrix multiplication is by taking a series of dot products: the 1st column of $\textit{A}$ times the 1st row of $\textit{B}$, the 2nd column of $\textit{A}$ times the 2nd row of $\textit{B}$, until the $n_{th}$ column of $\textit{A}$ times the $n_{th}$ row of $\textit{B}$. 

We define $\textit{A} \in \mathbb{R^{n\times p}} \cdot \textit{B} \in \mathbb{R^{n\times p}} = \textit{C} \in \mathbb{R^{m\times p}}$: 

$$
\textit{A}\cdot\textit{B}:=
\begin{bmatrix}
a_{11} & \cdots & a_{1n}\\
\vdots & \ddots & \vdots\\
a_{m1} & \cdots & a_{mn}
\end{bmatrix}
\begin{bmatrix}
b_{11} & \cdots & b_{1p}\\
\vdots & \ddots & \vdots\\
b_{n1} & \cdots & b_{np}
\end{bmatrix}=
\begin{bmatrix}
c_{11} & \cdots & c_{1p}\\
\vdots & \ddots & \vdots\\
c_{m1} & \cdots & c_{mp}
\end{bmatrix}
$$

A compact way to define the matrix-matrix product is:

$$
c_{ij} := \sum_{l=1}^n a_{il}b_{lj}, \text{  with   }i=1,...m, \text{ and}, j=1,...,p
$$

For instance

$$
\textit{A}\cdot\textit{B}=
\begin{bmatrix}
0 & 2\\
1 & 4
\end{bmatrix}
\begin{bmatrix}
1 & 3\\
2 & 1
\end{bmatrix}=
\begin{bmatrix}
1\times0 + 2\times2 & 3\times0 + 1\times2 \\
1\times1 + 2\times4 & 3\times1 + 1\times4
\end{bmatrix}=
\begin{bmatrix}
4 & 2\\
9 & 7
\end{bmatrix}
$$

In `NumPy`, we obtan the matrix-matrix product with the `@` operator or `dot` method: 


```python
A = np.array([[0,2],
              [1,4]])
B = np.array([[1,3],
              [2,1]])
```


```python
A @ B
```




    array([[4, 2],
           [9, 7]])




```python
np.dot(A, B)
```




    array([[4, 2],
           [9, 7]])



### Hadamard product

It is tempting to think in matrix-matrix multiplication as an element-wise operation, as multiplying each overlapping element of $\textit{A}$ and $\textit{B}$. *It is not*. Such operation is called **Hadamard product**. I'm introducing this to avoid confusion. The Hadamard product is defined as 

$$a_{ij} \cdot b_{ij} := c_{ij}$$

For instance:

$$
\textit{A}\odot\textit{B}=
\begin{bmatrix}
0 & 2\\
1 & 4
\end{bmatrix}
\begin{bmatrix}
1 & 3\\
2 & 1
\end{bmatrix}=
\begin{bmatrix}
0\times1 & 2\times3\\
1\times2 & 4\times 1\\
\end{bmatrix}=
\begin{bmatrix}
0 & 6\\
2 & 4\\
\end{bmatrix}
$$

In `numpy`, we compute the Hadamard product with the `*` operator or `multiply` method:


```python
A = np.array([[0,2],
              [1,4]])
B = np.array([[1,3],
              [2,1]])
```


```python
A * B
```




    array([[0, 6],
           [2, 4]])




```python
np.multiply(A, B)
```




    array([[0, 6],
           [2, 4]])



## Matrices as systems of linear equations

I introduced the idea of systems of linear equations as a way to figure out the right combination of linear segments to obtain an outcome. I did this in the context of vectors, now we can extend this to the context of matrices. 

Matrices are ideal to represent systems of linear equations. Consider the matrix $M$ and vectors $w$ and $y$ in $\in \mathbb{R}^3$. We can set up a system of linear equations as $Mw = y$ as:

$$
\begin{bmatrix}
m_{11} & m_{12} & m_{13} \\
m_{21} & m_{22} & m_{23} \\
m_{31} & m_{32} & m_{33} \\
\end{bmatrix}
\begin{bmatrix}
w_{1} \\
w_{2} \\
w_{3}
\end{bmatrix}=
\begin{bmatrix}
y_{1} \\
y_{2} \\
y_{3}
\end{bmatrix}
$$

This is equivalent to:
$$
\begin{matrix}
m_{11}w_{1} + m_{12}w_{2} + m_{13}w_{3} =y_{1} \\
m_{21}w_{1} + m_{22}w_{2} + m_{23}w_{3} =y_{2} \\
m_{31}w_{1} + m_{32}w_{2} + m_{33}w_{3} =y_{3}
\end{matrix}
$$

Geometrically, the solution for this representation equals to plot a **set of planes in 3-dimensional space**, one for each equation, and to find the segment where the planes intersect.

**Fig. 10: Visualiation system of equations as planes**


<img src="/assets/post-10/b-planes-intersection.svg">


An alternative way, which I personally prefer to use, is to represent the system as a **linear combination of the column vectors times a scaling term**:

$$
w_1
\begin{bmatrix}
m_{11}\\
m_{21}\\
m_{31}
\end{bmatrix}+
w_2
\begin{bmatrix}
m_{12}\\
m_{22}\\
m_{32}
\end{bmatrix}+
w_3
\begin{bmatrix}
m_{13}\\
m_{23}\\
m_{33}
\end{bmatrix}=
\begin{bmatrix}
y_{1} \\
y_{2} \\
y_{3}
\end{bmatrix}
$$

Geometrically, the solution for this representation equals to plot a set of **vectors in 3-dimensional** space, one for each column vector, then scale them by $w_i$ and add them up, tip to tail, to find the resulting vector $y$.

**Fig. 11: Visualiation system of equations as linear combination of vectors**


<img src="/assets/post-10/b-vectors-combination.svg">

