.. highlight:: python3

*********************************
Cake Eating II: Numerical Methods
*********************************

.. contents:: :depth: 2


In addition to what's in Anaconda, this lecture will require the following library:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade interpolation



Overview
========


In this lecture we continue the study of :doc:`the cake eating problem
<cake_eating_problem>`.

The aim of this lecture is to solve the problem using numerical
methods.

At first this might appear unnecessary, since we already obtained the optimal
policy analytically.

However, the cake eating problem is too simple to be useful without
modifications, and once we start modifying the problem, numerical methods become essential.

Hence it makes sense to introduce numerical methods now, and test them on this
simple problem.

Because we know the analytical solution, we can confirm that the numerical
methods are sound.

This will give us confidence in the methods before we shift to generalizations.

We will use the following imports:

.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

    from interpolation import interp
    from scipy.optimize import minimize_scalar, bisect



Reviewing The Model
===================

You might like to :doc:`review the details <cake_eating_problem>` before we start.

Recall in particular that the Bellman equation is

.. math::
    :label: bellman

    v(x) = \max_{0\leq c \leq x} \{u(c) + \beta v(x-c)\}
    \quad \text{for all } x \geq 0.

We found an analytical solution of the form 

.. math::
    v^*(x) = \left(1-\beta^{\frac{1}{\gamma}}\right)^{-\gamma} u(x)

where :math:`u` is the CRRA utility function.

Let's start by trying to obtain this analytical solution numerically.


Value Function Iteration
========================

The approach we will take is called **value function iteration**, which is a
form of **successive approximation** and was discussed in our :doc:`lecture on job search <mccall_model>`.

The basic idea is:

1. Take an arbitary intial guess of :math:`v`.

2. Obtain an update :math:`w` defined by 

    .. math::
        w(x) = \max_{0\leq c \leq x} \{u(c) + \beta v(x-c)\}

3. Stop if :math:`w` is approximately equal to :math:`v`, otherwise set
   :math:`v=w` and go back to step 2.

Let's write this a bit more mathematically.

The Bellman Operator
--------------------

We introduce the **Bellman operator** :math:`T` that takes a function `v` as an
argument and returns a new function :math:`Tv` defined by.

.. math::

    Tv(x) = \max_{0 \leq c \leq x} \{u(c) + \beta v(x - c)\}

From :math:`v` we get :math:`Tv`, and applying :math:`T` to this yields
:math:`T^2 v := T (Tv)` and so on.

This is called **iterating with the Bellman operator** from initial guess
:math:`v`.

As we discuss in more detail in later lectures, one can use Banach's
contraction mapping theorem to prove that the sequence of functions :math:`T^n
v` converges to the solution to the Bellman equation.



Fitted Value Function Iteration
-------------------------------

Both consumption :math:`c` and the state variable :math:`x` are continous. 

This causes complications when it comes to numerical work.

For example, we need to store each function :math:`T^n v` in order to
compute the next iterate :math:`T^{n+1} v`.

But this means we have to store :math:`T^n v(x)` at infinitely many :math:`x`, which is, in general, impossible.

To circumvent this issue we will use fitted value function iteration, as
discussed previously in :doc:`one of the lectures <mccall_fitted_vfi>` on job
search.

The process looks like this:

#. Begin with an array of values :math:`\{ v_0, \ldots, v_I \}`  representing
   the values of some initial function :math:`v` on the grid points :math:`\{ x_0, \ldots, x_I \}`.
#. Build a function :math:`\hat v` on the state space :math:`\mathbb R_+` by
   linear interpolation, based on these data points.
#. Obtain and record the value :math:`T \hat v(x_i)` on each grid point
   :math:`x_i` by repeatedly solving the maximization problem in the Bellman
   equation.
#. Unless some stopping condition is satisfied, set
   :math:`\{ v_0, \ldots, v_I \} = \{ T \hat v(x_0), \ldots, T \hat v(x_I) \}` and go to step 2.

In step 2 we'll use continuous piecewise linear interpolation.



Implementation
--------------

The ``maximize`` function below is a small helper function that converts a
SciPy minimization routine into a maximization routine.

.. code-block:: python3

    def maximize(g, a, b, args):
        """
        Maximize the function g over the interval [a, b].

        We use the fact that the maximizer of g on any interval is
        also the minimizer of -g.  The tuple args collects any extra
        arguments to g.

        Returns the maximal value and the maximizer.
        """

        objective = lambda x: -g(x, *args)
        result = minimize_scalar(objective, bounds=(a, b), method='bounded')
        maximizer, maximum = result.x, -result.fun
        return maximizer, maximum

We'll store the parameters :math:`\beta` and :math:`\gamma` in a 
class called ``CakeEating``. 

The same class will also provide a method called `state_action_value` that
returns the value of a consumption choice given a particular state and guess
of :math:`v`.

.. code-block:: python3

    class CakeEating:

        def __init__(self,
                     β=0.96,           # discount factor
                     γ=1.5,            # degree of relative risk aversion
                     x_grid_min=1e-3,  # exclude zero for numerical stability
                     x_grid_max=2.5,   # size of cake
                     x_grid_size=120):

            self.β, self.γ = β, γ

            # Set up grid
            self.x_grid = np.linspace(x_grid_min, x_grid_max, x_grid_size)

        # Utility function
        def u(self, c):

            γ = self.γ

            if γ == 1:
                return np.log(c)
            else:
                return (c ** (1 - γ)) / (1 - γ)

        # first derivative of utility function
        def u_prime(self, c):

            return c ** (-self.γ)

        def state_action_value(self, c, x, v_array):
            """
            Right hand side of the Bellman equation given x and c.
            """

            u, β = self.u, self.β
            v = lambda x: interp(self.x_grid, v_array, x)

            return u(c) + β * v(x - c)


We now define the Bellman operation:

.. code-block:: python3

    def T(v, ce):
        """
        The Bellman operator.  Updates the guess of the value function.

        * ce is an instance of CakeEating
        * v is an array representing a guess of the value function

        """
        v_new = np.empty_like(v)

        for i, x in enumerate(ce.x_grid):
            # Maximize RHS of Bellman equation at state x
            v_new[i] = maximize(ce.state_action_value, 1e-10, x, (x, v))[1]

        return v_new

After defining the Bellman operator, we are ready to solve the model.

Let's start by creating a ``CakeEating`` instance using the default parameterization.

.. code-block:: python3

    ce = CakeEating()

Now let's see the iteration of the value function in action. We choose an
intial guess whose value is :math:`0` for every :math:`y` grid point. 

We should see that the value functions converge to a fixed point as we apply
Bellman operations.

.. code-block:: python3

    x_grid = ce.x_grid
    v = ce.u(x_grid)       # Initial guess
    n = 12                 # Number of iterations

    fig, ax = plt.subplots()

    ax.plot(x_grid, v, color=plt.cm.jet(0),
            lw=2, alpha=0.6, label='Initial guess')

    for i in range(n):
        v = T(v, ce)  # Apply the Bellman operator
        ax.plot(x_grid, v, color=plt.cm.jet(i / n), lw=2, alpha=0.6)

    ax.legend()
    ax.set_ylabel('value', fontsize=12)
    ax.set_xlabel('cake size $x$', fontsize=12)
    ax.set_title('Value function iterations')

    plt.show()

We can define a wrapper function ``compute_value_function`` which does the value function iterations
until some convergence conditions are satisfied and then return a converged value function.

.. code-block:: python3

    def compute_value_function(ce,
                               tol=1e-4,
                               max_iter=1000,
                               verbose=True,
                               print_skip=25):

        # Set up loop
        v = np.zeros(len(ce.x_grid)) # Initial guess
        i = 0
        error = tol + 1

        while i < max_iter and error > tol:
            v_new = T(v, ce)

            error = np.max(np.abs(v - v_new))
            i += 1

            if verbose and i % print_skip == 0:
                print(f"Error at iteration {i} is {error}.")

            v = v_new

        if i == max_iter:
            print("Failed to converge!")

        if verbose and i < max_iter:
            print(f"\nConverged in {i} iterations.")

        return v_new


Now let's call it --- note that it takes a little while to run.

.. code-block:: python3

    v = compute_value_function(ce)

Now we can plot and see what the converged value function looks like. 

.. code-block:: python3

    fig, ax = plt.subplots()

    ax.plot(x_grid, v, label='Approximate value function')
    ax.set_ylabel('$V(y)$', fontsize=12)
    ax.set_xlabel('$y$', fontsize=12)
    ax.set_title('Value function')
    ax.legend()
    plt.show()



The function defined below computes the analytical solution of a given ``CakeEating`` instance.

.. code-block:: python3

    def v_star(ce):

        β, γ = ce.β, ce.γ
        x_grid = ce.x_grid
        u = ce.u

        a = β ** (1 / γ)
        x = 1 - a
        z = u(x_grid)

        return z / x ** γ

.. code-block:: python3

    v_analytical = v_star(ce)

.. code-block:: python3

    fig, ax = plt.subplots()

    ax.plot(x_grid, v_analytical, label='analytical solution')
    ax.plot(x_grid, v, label='numerical solution')
    ax.set_ylabel('$V(y)$', fontsize=12)
    ax.set_xlabel('$y$', fontsize=12)
    ax.legend()
    ax.set_title('Comparison between analytical and numerical value functions')
    plt.show()

The quality of approximation is reasonably good, although less so near the
lower boundary.

The issue here is that the utility function and hence value function is very
steep in this region and hence hard to approximate with linear interpolation.

Let's see how this plays out in terms of computing the optimal policy.



Policy Function
---------------

In the :doc:`first lecture on cake eating <cake_eating_problem>`, the optimal
consumption policy was shown to be

.. math::
    \sigma^*(x) = \left(1-\beta^\frac{1}{\gamma}\right) x

Let's see if our numerical results lead to something similar.

Our numerical strategy will be to compute 

.. math::
    \sigma(x) = \arg \max_{0 \leq c \leq x} \{u(c) + \beta v(x - c)\}

on a grid of :math:`x` points and then interpolate.

For :math:`v` we will use the approximation of the value function we obtained
above.

Here's the function:

.. code-block:: python3

    def σ(ce, v):
        """
        The optimal policy function. Given the value function,
        it finds optimal consumption in each state.

        * ce is an instance of CakeEating
        * v is a value function array

        """
        c = np.empty_like(v)

        for i in range(len(ce.x_grid)):
            y = ce.x_grid[i]
            # Maximize RHS of Bellman equation at state y
            c[i] = maximize(ce.state_action_value, 1e-10, y, (y, v))[0]

        return c

Now let's pass the approximate value function and compute optimal consumption:

.. code-block:: python3

    c = σ(ce, v)  

.. code-block:: python3

    fig, ax = plt.subplots()

    ax.plot(x_grid, c)
    ax.set_ylabel('$\sigma(y)$')
    ax.set_xlabel('$y$')
    ax.set_title('Optimal policy')
    plt.show()

.. _pol_an:


Let's compare it to the true analytical solution.

.. code-block:: python3

    def c_star(ce):

        β, γ = ce.β, ce.γ
        x_grid = ce.x_grid

        return (1 - β ** (1/γ)) * x_grid

.. code-block:: python3

    c_analytical = c_star(ce)

.. code-block:: python3

    fig, ax = plt.subplots()

    ax.plot(ce.x_grid, c_analytical, label='analytical')
    ax.plot(ce.x_grid, c, label='Numerical')
    ax.set_ylabel('$\sigma(y)$')
    ax.set_xlabel('$y$')
    ax.legend()
    ax.set_title('Comparison between analytical and numerical optimal policies')
    plt.show()


The fit is not perfect but quite good.

We can improve it further by increasing the grid size or reducing the
error tolerance in the value function iteration routine.

Of course both changes will lead to a longer compute time.

Another possibility is to use an alternative algorithm, which offers the
possibility of faster compute time and, at the same time, more accuracy.

We explore this next.



Time Iteration
==============

Now let's look at a different strategy to compute the optimal policy.

Recall that the optimal policy satisfies the Euler equation 

.. math::
    :label: euler

    u' (\sigma(x)) = \beta u' ( \sigma( (x - \sigma(x)) ))
    \quad \text{for all } x > 0

Computationally, we can start with any initial guess of
:math:`\sigma_0` and now choose :math:`c` to solve

.. math::

    u^{\prime}( c ) = \beta u^{\prime} (\sigma_0(y - c))

Chosing :math:`c` that satisfies this equation at all :math:`x > 0` produces a function of :math:`x`.

Call this new function :math:`\sigma_1`, treat it and the new guess and
repeat.

This is called **time iteration**.

As with value function iteration, we can view the update step as action of an
operator, this time denoted by :math:`K`.

* In particular, :math:`K\sigma` is the policy updated from :math:`\sigma`
  using the procedure just described.

The main advantage of time iteration relative to value function iteration is that it operates
in policy space rather than value function space.

This is helpful because policy functions have less curvature, at least for the
current example, and hence are easier to approximate.

In the exercises you are asked to implement time iteration and compare it to
value function iteration.

You should find that the method is faster and more accurate.

The intuition behind this is that we are using more information --- in this
case, the first order conditions.


Exercises
=========


Exercise 1
------------

Try the following modification of the problem.

Instead of the cake size changing according to :math:`x_{t+1} = x_t - c_t`,
let it change according to 

.. math::
    x_{t+1} = (x_t - c_t)^{\alpha}

where :math:`\alpha` is a parameter satisfying :math:`0 < \alpha < 1`.

(We will see this kind of update rule when we study optimal growth models.)

Make the required changes to value function iteration code and plot the value and policy functions. 

Try to reuse as much code as possible.


Exercise 2
----------

Implement time iteration, returning to the original case (i.e., dropping the
modification in the exercise above).




Solutions
==========


Exercise 1
-----------

We need to create a class to hold our primitives and return the right hand side of the bellman equation.

We will use `inheritance <https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)>`__ to maximize code reuse.

.. code-block:: python3

    class OptimalGrowth(CakeEating):
        """
        A subclass of CakeEating that adds the parameter α and overrides
        the state_action_value method.
        """

        def __init__(self,
                     β=0.96,           # discount factor
                     γ=1.5,            # degree of relative risk aversion
                     α=0.4,            # productivity parameter
                     x_grid_min=1e-3,  # exclude zero for numerical stability
                     x_grid_max=2.5,   # size of cake
                     x_grid_size=120):

            self.α = α 
            CakeEating.__init__(self, β, γ, x_grid_min, x_grid_max, x_grid_size)

        def state_action_value(self, c, x, v_array):
            """
            Right hand side of the Bellman equation given x and c.
            """

            u, β, α = self.u, self.β, self.α
            v = lambda x: interp(self.x_grid, v_array, x)

            return u(c) + β * v((x - c)**α)

.. code-block:: python3

    og = OptimalGrowth()

Here's the computed value function.

.. code-block:: python3

    v = compute_value_function(og, verbose=False)

    fig, ax = plt.subplots()

    ax.plot(x_grid, v, lw=2, alpha=0.6)
    ax.set_ylabel('value', fontsize=12)
    ax.set_xlabel('state $x$', fontsize=12)

    plt.show()

Here's the computed policy, combined with the solution we derived above for
the standard cake eating case :math:`\alpha=1`.

.. code-block:: python3

    c_new = σ(og, v)

    fig, ax = plt.subplots()

    ax.plot(ce.x_grid, c_analytical, label='$\\alpha=1$ solution')
    ax.plot(ce.x_grid, c_new, label=f'$\\alpha={og.α}$ solution')

    ax.set_ylabel('consumption', fontsize=12)
    ax.set_xlabel('$x$', fontsize=12)

    ax.legend(fontsize=12)

    plt.show()


Consumption is higher when :math:`\alpha < 1` because, at least for large :math:`x`, the return to savings is lower.




Exercise 2
----------

Here's one way to implement time iteration.


.. code-block:: python3

    def K(σ_array, ce):
        """
        The policy function operator. Given the policy function,
        it updates the optimal consumption using Euler equation.

        * σ_array is an array of policy function values on the grid
        * ce is an instance of CakeEating

        """

        u_prime, β, x_grid = ce.u_prime, ce.β, ce.x_grid
        σ_new = np.empty_like(σ_array)

        σ = lambda x: interp(x_grid, σ_array, x)

        def euler_diff(c, x):
            return u_prime(c) - β * u_prime(σ(x - c))

        for i, x in enumerate(x_grid):

            # handle small x separately --- helps numerical stability
            if x < 1e-12:
                σ_new[i] = 0.0

            # handle other x 
            else:
                σ_new[i] = bisect(euler_diff, 1e-10, x - 1e-10, x)

        return σ_new


.. code-block:: python3

    def iterate_euler_equation(ce,
                               max_iter=500,
                               tol=1e-5,
                               verbose=True,
                               print_skip=25):

        x_grid = ce.x_grid

        σ = np.copy(x_grid)        # initial guess

        i = 0
        error = tol + 1
        while i < max_iter and error > tol:

            σ_new = K(σ, ce)

            error = np.max(np.abs(σ_new - σ))
            i += 1

            if verbose and i % print_skip == 0:
                print(f"Error at iteration {i} is {error}.")

            σ = σ_new

        if i == max_iter:
            print("Failed to converge!")

        if verbose and i < max_iter:
            print(f"\nConverged in {i} iterations.")

        return σ 

.. code-block:: python3

    ce = CakeEating(x_grid_min=0.0)
    c_euler = iterate_euler_equation(ce)

.. code-block:: python3

    fig, ax = plt.subplots()

    ax.plot(ce.x_grid, c_analytical, label='analytical solution')
    ax.plot(ce.x_grid, c_euler, label='time iteration solution')

    ax.set_ylabel('consumption')
    ax.set_xlabel('$x$')
    ax.legend(fontsize=12)

    plt.show()


