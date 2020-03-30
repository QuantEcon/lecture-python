

.. highlight:: python3

**********************************************
Cake Eating I: Introduction to Optimal Saving
**********************************************

.. contents:: :depth: 2



Overview
========


In this lecture we introduce a simple "cake eating" problem.

The intertemporal problem is: how much to enjoy today and how much to leave
for the future?

All though the topic sounds trivial, this kind of trade-off between current
and future utility is at the heart of many savings and consumption problems.

Once we master the ideas in this simple environment, we will apply them to
progressively more challenging---and useful---problems.

The main tool we will use to solve the cake eating problem is dynamic programming.

This topic is an excellent way to build dynamic programming skills.

Although not essential, readers will find it helpful to review the following
lectures before reading this one:

* The :doc:`shortest paths lecture <short_path>`
* The :doc:`basic McCall model <mccall_model>`
* The :doc:`McCall model with separation <mccall_model_with_separation>`
* The :doc:`McCall model with separation and a continuous wage distribution <mccall_fitted_vfi>` 

In what follows, we require the following imports:


.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline



The Model
==================

We operating on a infinite time horizon :math:`t=0, 1, 2, 3..`

At :math:`t=0` the agent is given a complete cake with size :math:`\bar x`.

Let :math:`x_t` denote the size of the cake at the beginning of each period,
so that, in particular, :math:`x_0=\bar x`.

We choose how much of the cake to eat in any given period :math:`t`.

After choosing to consume :math:`c_t` of the cake in period :math:`t` there is

.. math::
    x_{t+1} = x_t - c_t 

left in period :math:`t+1`.


Consuming quantity :math:`c` of the cake gives current utility :math:`u(c)`.

We adopt the CRRA utility function

.. math::
    u(c) = \frac{c^{1-\gamma}}{1-\gamma} \qquad (\gamma \neq 1)
    :label: crra_utility

In Python this is

.. code-block:: python3

    def u(c, γ):

        return c**(1 - γ) / (1 - γ)


Future cake consumption utility is discounted according to :math:`\beta\in(0, 1)`.

In particular, consumption of :math:`c` units :math:`t` periods hence has present value :math:`\beta^t u(c)`

The agent's problem can be written as

.. math::
    \max_{\{c_t\}} \sum_{t=0}^\infty \beta^t u(c_t)
    :label: cake_objective

subject to

.. math::
    x_{t+1} = x_t - c_t 
    \quad \text{and} \quad
    0\leq c_t\leq x_t
    :label: cake_feasible

for all :math:`t`.


A consumption path :math:`\{c_t\}` satisfying :eq:`cake_feasible` where
:math:`x_0 = \bar x` is called **feasible**.

In this problem, the following terminology is standard:

* :math:`x_t` is called the **state variable**
* :math:`c_t` is called the **control variable** or the **action**
* :math:`\beta` and :math:`\gamma` are **parameters**



Trade-Off
---------

The key trade-off in the cake-eating problem is this:

* Delaying consumption is costly because of the discount factor.

* But delaying some consumption is also attractive because :math:`u` is concave.


The concavity of :math:`u` implies that the consumer gains value from
*consumption smoothing*, which means spreading consumption out over time.

This is because concavity implies diminishing marginal utility---a progressively smaller gain in utility for each additional spoonful of cake consumed within one period.


Intuition
---------

The reasoning given above suggests that the discount factor :math:`\beta` and the curvature parameter :math:`\gamma` will play a key role in determining the rate of consumption.

Here's an educated guess as to what impact these parameters will have.

First, higher :math:`\beta` implies less discounting, and hence more patience, which should reduce the rate of consumption.

Second, higher :math:`\gamma` implies that marginal utility :math:`u'(c) =
c^{-\gamma}` falls faster with :math:`c`.

This suggests more smoothing, and hence a lower rate of consumption.

In summary, we expect the rate of consumption to be *decreasing in both
parameters*.

Let's see if this is true.




The Value Function
==================

The first step of our dynamic programming treatment is to obtain the Bellman
equation.

The next step is to use it to calculate the solution.


The Bellman Equation
--------------------

To this end, we let :math:`v(x)` be maximum lifetime utility attainable from
the current time when :math:`x` units of cake are left.

That is,

.. math::
    v(x) = \max \sum_{t=0}^{\infty} \beta^t u(c_t) 
    :label: value_fun

where the maximization is over all paths :math:`\{ c_t \}` that are feasible
from :math:`x_0 = x`.

At this point, we do not have an expression for :math:`v`, but we can still
make inferences about it.

For example, as was the case with the :doc:`McCall model <mccall_model>`, the
value function will satisfy a version of the *Bellman equation*.

In the present case, this equation states that :math:`v` satisfies 

.. math::
    :label: bellman

    v(x) = \max_{0\leq c \leq x} \{u(c) + \beta v(x-c)\}
    \quad \text{for any given } x \geq 0.

The intuition here is essentially the same it was for the McCall model.

Suppose that the current size of the cake is :math:`x`.

choosing :math:`c` optimally means trading off current vs future rewards.

Current rewards from choice :math:`c` are just :math:`u(c)`.

Future rewards, measured from next period and assuming optimal behavior, are :math:`v(x-c)`.

These are the two terms on the right hand side of :eq:`bellman`, after suitable discounting.

If :math:`c` is chosen optimally using this trade off strategy, then we obtain maximal
lifetime rewards from our current state :math:`x`.

Hence, :math:`v(x)` equals the right hand side of :eq:`bellman`, as claimed.


An Analytical Solution
----------------------

It has been shown that, with :math:`u` as the CRRA utility function in
:eq:`crra_utility`, the function

.. math::
    v^*(x_t) = \left(1-\beta^\frac{1}{\gamma}\right)^{-\gamma}u(x_t)
    :label: crra_vstar

solves the Bellman equation and hence is equal to the value function.

You are asked to confirm that this is true in the exercises below.

The solution :eq:`crra_vstar` depends heavily on the CRRA utility function.

In fact, if we move away from CRRA utility, usually there is no analytical
solution at all.

In other words, beyond CRRA utility, we know that the value function still
satisfies the Bellman equation, but we do not have a way of writing it
explicitly, as a function of the state variable and the parameters.

We will deal with that situation numerically when the time comes.

Here is a Python representation of the value function:



.. code-block:: python3

    def v_star(x, β, γ):

        return (1 - β**(1 / γ))**(-γ) * u(x, γ)


And here's a figure showing the function for fixed parameters:


.. code-block:: python3

    β, γ = 0.95, 1.2
    x_grid = np.linspace(0.1, 5, 100)

    fig, ax = plt.subplots()

    ax.plot(x_grid, v_star(x_grid, β, γ), label='value function')

    ax.set_xlabel('$x$', fontsize=12)
    ax.legend(fontsize=12)

    plt.show()



The Optimal Policy
==================

Now that we have the value function, it is straightforward to calculate the
optimal action at each state.

At state :math:`x`, we should choose :math:`c` as the value the maximizes the
right hand side of the Bellman equation :eq:`bellman`.

.. math::
    c^*_t = \sigma(x_t) = \arg \max_{c_t} \{u(c_t) + \beta v(x_t - c_t)\}

We can think of this optimal choice as a function of the state :math:`x`, in
which case we call it the **optimal policy**.

We will denote the optimal policy by :math:`\sigma^*`, so that

.. math::
    \sigma^*(x) = \arg \max_{c} \{u(c) + \beta v(x - c)\}
    \quad \text{for all } x

If we plug the analytical expression :eq:`crra_vstar` for the value function
into the right hand side and compute the optimum, we find that 

.. math::
    \sigma^*(x) = \left(1-\beta^\frac{1}{\gamma}\right)x
    :label: crra_opt_pol




Now let's recall our intuition on the impact of parameters.

We guessed that the consumption rate would be decreasing in both parameters.

This is in fact the case, as can be seen from :eq:`crra_opt_pol`.

Here's some plots that illustrate.


.. code-block:: python3

    def c_star(x, β, γ):

        return (1 - β ** (1/γ)) * x

Continuing with the values for :math:`\beta` and :math:`\gamma` used above, the
plot is

.. code-block:: python3

    fig, ax = plt.subplots()
    ax.plot(x_grid, c_star(x_grid, β, γ), label='default parameters')
    ax.plot(x_grid, c_star(x_grid, β + 0.02, γ), label='higher $\\beta$')
    ax.plot(x_grid, c_star(x_grid, β, γ + 0.2), label='higher $\gamma$')
    ax.set_ylabel('$\sigma(x)$')
    ax.set_xlabel('$x$')
    ax.legend()

    plt.show()


The Euler Equation
==================

In the discussion above we have provided a complete solution to the cake
eating problem in the case of CRRA utility.

There is in fact another way to solve for the optimal policy, based on the
so-called **Euler equation**.

Although we already have a complete solution, now is a good time to study the
Euler equation.

This is because, for more difficult problems, this equation
provides key insights that are hard to obtain by other methods.



Statement and Implications
--------------------------

The Euler equation for the present problem can be stated as 

.. math::
    :label: euler

    u^{\prime} (c^*_{t})=\beta u^{\prime}(c^*_{t+1})

This is necessary condition for the optimal path.  

It says that, along the optimal path, marginal rewards are equalized across time, after appropriate discounting.

This makes sense: optimality is obtained by smoothing consumption up to the
point where no marginal gains remain.

We can also state the Euler equation in terms of the policy function.

A **feasible consumption policy** is a map :math:`y \mapsto \sigma(y)`
satisfying :math:`0 \leq \sigma(y) \leq (y)`.

The last restriction says that we cannot consume more than the remaining
quantity of cake.

A feasible consumption policy :math:`\sigma` is said to **satisfy the Euler equation** if, for
all :math:`y > 0`,

.. math::
    :label: euler_pol

    u^{\prime}( \sigma(x) )
    = \beta u^{\prime} (\sigma(x - \sigma(x)))

Evidently :eq:`euler_pol` is just the policy equivalent of :eq:`euler`.

It turns out that a feasible policy is optimal if and
only if it satisfies the Euler equation.

In the exercises, you are asked to verify that the optimal policy
:eq:`crra_opt_pol` does indeed satisfy this functional equation.

.. note::
    A **functional equation** is an equation where the unknown object is a function.

For a proof of sufficiency of the Euler equation in a very general setting,
see proposition 2.2 of :cite:`ma2020income`.

The following arguments focus on necessity, explaining why an optimal path or 
policy should satisfy the Euler equation.




Derivation I: A Perturbation Approach
-------------------------------------

Let's write :math:`c` as a shorthand for consumption path :math:`\{c_t\}_{t=0}^\infty`.

The overall cake-eating maximization problem can be written as

.. math::
    \max_{c \in F} U(c) 
    \quad \text{where } U(c) := \max_{\{c_t\}} \sum_{t=0}^\infty \beta^t u(c_t)

and :math:`F` is the set of feasible consumption paths.

We know that differentiable functions have a zero gradient at a maximizer.

So the optimal path :math:`c^* := \{c^*_t\}_{t=0}^\infty` must satisfy
:math:`U'(c^*) = 0`.

.. note::
    If you insist on knowing exactly how the derivative :math:`U'(c^*)` is defined, 
    given that the argument
    :math:`c^*` is a vector of infinite length, you can start by learning about 
    `Gateaux derivatives <https://en.wikipedia.org/wiki/Gateaux_derivative>`__,
    although such knowledge is not assumed in what follows.

In other words, the rate of change in :math:`U` must be zero for any
infinitesimally small (and feasible) perturbation away from the optimal path.

So consider a feasible perturbation that reduces consumption at time :math:`t` to 
:math:`c^*_t - h`
and increases it in the next period to :math:`c^*_{t+1} + h`.

Consumption does not change in any other period.

We call this perturbed path :math:`c^h`.

By the preceding argument about zero gradients, we have

.. math::
    \lim_{h \to 0} \frac{U(c^h) - U(c^*)}{h} = U'(c^*) = 0


Recalling that consumption only changes at :math:`t` and :math:`t+1`, this
becomes

.. math::
    \lim_{h \to 0} 
    \frac{\beta^t u(c^*_t - h) + \beta^{t+1} u(c^*_{t+1} + h) 
          - \beta^t u(c^*_t) - \beta^{t+1} u(c^*_{t+1}) }{h} = 0

After rearranging, the same expression can be written as

.. math::
    \lim_{h \to 0} 
        \frac{u(c^*_t - h) - u(c^*_t) }{h}
    + \lim_{h \to 0} 
        \frac{ \beta u(c^*_{t+1} + h) - u(c^*_{t+1}) }{h} = 0

or, taking the limit,

.. math::
    - u'(c^*_t) + \beta u'(c^*_{t+1}) = 0

This is just the Euler equation.


Derivation II: Using the Bellman Equation
------------------------------------------

Another way to derive the Euler equation is to use the Bellman equation :eq:`bellman`. 

Taking the derivative on the right hand side of the Bellman equation with
respect to :math:`c` and setting it to zero, we get

.. math::
    :label: bellman_FOC

    u^{\prime}(c)=\beta v^{\prime}(x - c)

To obtain :math:`v^{\prime}(x - c)`, we set
:math:`g(c,x) = u(c) + \beta v(x - c)`, so that, at the optimal choice of
consumption, 

.. math::
    :label: bellman_equality

    v(x) = g(c,x)

Differentiating both sides while acknowledging that the maximizing consumption will depend
on :math:`x`, we get

.. math::
    v' (x) = 
    \frac{\partial }{\partial c} g(c,x) \frac{\partial c}{\partial x}
     + \frac{\partial }{\partial x} g(c,x)
    

When :math:`g(c,x)` is maximized at :math:`c`, we have :math:`\frac{\partial }{\partial c} g(c,x) = 0`.

Hence the derivative simplifies to

.. math::
    v' (x) = 
    \frac{\partial g(c,x)}{\partial x}
    = \frac{\partial }{\partial x} \beta v(x - c)
    = \beta v^{\prime}(x - c)
    :label: bellman_envelope


(This argument is an example of the `Envelope Theorem <https://en.wikipedia.org/wiki/Envelope_theorem>`__.) 


But now an application of :eq:`bellman_FOC` gives

.. math::
    :label: bellman_v_prime

    u^{\prime}(c) = v^{\prime}(x)

Thus, the derivative of the value function is equal to marginal utility.

Combining this fact with :eq:`bellman_envelope` recovers the Euler equation.


Exercises
=========

Exercise 1
------------

How does one obtain the expressions for the value function and optimal policy
given in :eq:`crra_vstar` and :eq:`crra_opt_pol` respectively?

The first step is to make a guess of the functional form for the consumption
policy.

So suppose that we do not know the solutions and start with a guess that the
optimal policy is linear.

In other words, we conjecture that there exists a positive :math:`\theta` such that setting :math:`c_t^*=\theta x_t` for all :math:`t` produces an optimal path.

Starting from this conjecture, try to obtain the solutions :eq:`crra_vstar` and :eq:`crra_opt_pol`.


In doing so, you will need to use the definition of the value function and the
Bellman equation.


Solutions
==========


Exercise 1
-----------

We start with the conjecture :math:`c_t^*=\theta x_t`, which leads to a path
for the state variable (cake size) given by 

.. math::
    x_{t+1}=x_t(1-\theta)

Then :math:`x_t = x_{0}(1-\theta)^t` and hence


.. math::

    \begin{aligned}
    v(x_0) 
       & = \sum_{t=0}^{\infty} \beta^t u(\theta x_t)\\
       & = \sum_{t=0}^{\infty} \beta^t u(\theta x_0 (1-\theta)^t ) \\
       & = \sum_{t=0}^{\infty} \theta^{1-\gamma} \beta^t (1-\theta)^{t(1-\gamma)} u(x_0) \\
       & = \frac{\theta^{1-\gamma}}{1-\beta(1-\theta)^{1-\gamma}}u(x_{0})
    \end{aligned}

From the Bellman equation, then,

.. math::
    \begin{aligned}
        v(x) & = \max_{0\leq c\leq x}
            \left\{
                u(c) + 
                \beta\frac{\theta^{1-\gamma}}{1-\beta(1-\theta)^{1-\gamma}}\cdot u(x-c)
            \right\} \\
             & = \max_{0\leq c\leq x}
                \left\{
                    \frac{c^{1-\gamma}}{1-\gamma} + 
                    \beta\frac{\theta^{1-\gamma}}
                    {1-\beta(1-\theta)^{1-\gamma}}
                    \cdot\frac{(x-c)^{1-\gamma}}{1-\gamma}
                \right\}
    \end{aligned}

From the first order condition, we obtain

.. math::
    c^{-\gamma} + \beta\frac{\theta^{1-\gamma}}{1-\beta(1-\theta)^{1-\gamma}}\cdot(x-c)^{-\gamma}(-1) = 0

or

.. math::
    c^{-\gamma} = \beta\frac{\theta^{1-\gamma}}{1-\beta(1-\theta)^{1-\gamma}}\cdot(x-c)^{-\gamma}


With :math:`c = \theta x` we get

.. math::
    \left(\theta x\right)^{-\gamma} =  \beta\frac{\theta^{1-\gamma}}{1-\beta(1-\theta)^{1-\gamma}}\cdot(x(1-\theta))^{-
    \gamma}

Some rearrangement produces

.. math::
    \theta = 1-\beta^{\frac{1}{\gamma}}


This confirms our earlier expression for the optimal policy:

.. math::
    c_t^* = \left(1-\beta^{\frac{1}{\gamma}}\right)x_t


Substituting :math:`\theta` into the value function above gives.

.. math::
    v^*(x_t) = \frac{\left(1-\beta^{\frac{1}{\gamma}}\right)^{1-\gamma}}
    {1-\beta\left(\beta^{\frac{{1-\gamma}}{\gamma}}\right)} u(x_t) \\

Rearranging gives

.. math::
    v^*(x_t) = \left(1-\beta^\frac{1}{\gamma}\right)^{-\gamma}u(x_t)


Our claims are now verified.


