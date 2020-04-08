.. _ifp:

.. include:: /_static/includes/header.raw

.. highlight:: python3

******************************************************
:index:`The Income Fluctuation Problem I: Basic Model`
******************************************************

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon
  !pip install interpolation



Overview
========

Next, we study an optimal savings problem for an infinitely lived consumer---the "common ancestor" described in :cite:`Ljungqvist2012`, section 1.3.

This is an essential sub-problem for many representative macroeconomic models

* :cite:`Aiyagari1994`

* :cite:`Huggett1993`

* etc.

It is related to the decision problem in the :doc:`stochastic optimal growth
model <optgrowth>` and yet differs in important ways.

For example, the choice problem for the agent includes an additive income term that leads to an occasionally binding constraint.

Moreover, in this and the following lectures, we will inject more realisitic
features such as correlated shocks.

To solve the model we will use Euler equation based time iteration, similar to :doc:`this lecture <coleman_policy_iter>`.

This method turns out to be globally convergent under mild assumptions, even when utility is unbounded (both above and below).

We'll need the following imports:

.. code-block:: ipython

    import numpy as np
    from quantecon.optimize import brent_max, brentq
    from interpolation import interp
    from numba import njit, float64, jitclass
    import matplotlib.pyplot as plt
    %matplotlib inline
    from quantecon import MarkovChain


References
----------

Our presentation is a simplified version of :cite:`ma2020income`.

Other references include :cite:`Deaton1991`, :cite:`DenHaan2010`,
:cite:`Kuhn2013`, :cite:`Rabault2002`,  :cite:`Reiter2009`  and
:cite:`SchechtmanEscudero1977`.



The Optimal Savings Problem
===========================

.. index::
    single: Optimal Savings; Problem

Let's write down the model and then discuss how to solve it.

Set-Up
------

Consider a household that chooses a state-contingent consumption plan :math:`\{c_t\}_{t \geq 0}` to maximize

.. math::

    \mathbb{E} \, \sum_{t=0}^{\infty} \beta^t u(c_t)


subject to

.. math::
    :label: eqst

    a_{t+1} \leq  R(a_t - c_t)  + Y_{t+1},
    \qquad c_t \geq 0,
    \qquad a_t \geq 0
    \qquad t = 0, 1, \ldots

Here

* :math:`\beta \in (0,1)` is the discount factor

* :math:`a_t` is asset holdings at time :math:`t`, with borrowing constraint :math:`a_t \geq 0`

* :math:`c_t` is consumption

* :math:`Y_t` is non-capital income (wages, unemployment compensation, etc.)

* :math:`R := 1 + r`, where :math:`r > 0` is the interest rate on savings

The timing here is as follows:

#. At the start of period :math:`t`, the household chooses consumption
   :math:`c_t`.

#. Labor is supplied by the household throughout the period and labor income
   :math:`Y_{t+1}` is received at the end of period :math:`t`.

#. Financial income :math:`R(a_t - c_t)` is received at the end of period :math:`t`.

#. Time shifts to :math:`t+1` and the process repeats.


Non-capital income :math:`Y_t` is given by :math:`Y_t = y(Z_t)`, where
:math:`\{Z_t\}` is an exogeneous state process.

In particular, we take :math:`\{Z_t\}` to be a finite state Markov chain
taking values in :math:`\mathsf Z` with Markov matrix :math:`P`.

We further assume that

#. :math:`r > 0` and :math:`\beta R < 1`

#. :math:`u` is smooth, strictly increasing and strictly concave with :math:`\lim_{c \to 0} u'(c) = \infty` and :math:`\lim_{c \to \infty} u'(c) = 0`


The asset space is :math:`\mathbb R_+` and the state is the pair :math:`(a,z)
\in \mathsf S := \mathbb R_+ \times \mathsf Z`.

A *feasible consumption path* from :math:`(a,z) \in \mathsf S` is a consumption
sequence :math:`\{c_t\}` such that :math:`\{c_t\}` and its induced asset path :math:`\{a_t\}` satisfy

#. :math:`(a_0, z_0) = (a, z)`

#. the feasibility constraints in :eq:`eqst`, and

#. measurability, which means that :math:`c_t` is a function of random
   outcomes up to date :math:`t` but not after.

The meaning of the third point is just that consumption at time :math:`t`
cannot be a function of outcomes are yet to be observed.

In fact, for this problem, consumption can be chosen optimally by taking it to
be contingent only on the current state.

Optimality is defined below.


Value Function and Euler Equation
---------------------------------

The *value function* :math:`V \colon S \to \mathbb{R}` is defined by

.. math::
    :label: eqvf

    V(a, z) := \sup \, \mathbb{E}
    \left\{
    \sum_{t=0}^{\infty} \beta^t u(c_t)
    \right\}


where the supremum is overall feasible consumption paths from :math:`(a,z)`.

An *optimal consumption path* from :math:`(a,z)` is a feasible consumption path from :math:`(a,z)` that attains the supremum in :eq:`eqvf`.

To pin down such paths we can use a version of the Euler equation, which in the present setting is

.. math::
    :label: ee00

    u' (c_t)
    \geq \beta R \,  \mathbb{E}_t [ u'(c_{t+1}) ]

and

.. math::
    :label: ee01

    c_t < a_t 
    \; \implies \;
    u' (c_t) = \beta R \,  \mathbb{E}_t [ u'(c_{t+1}) ]

When :math:`c_t = a_t` we obviously have :math:`u'(c_t) = u'(a_t)`,

When :math:`c_t` hits the upper bound :math:`a_t`, the
strict inequality :math:`u' (c_t) > \beta R \,  \mathbb{E}_t [ u'(c_{t+1}) ]`
can occur because :math:`c_t` cannot increase sufficiently to attain equality.

(The lower boundary case :math:`c_t = 0` never arises at the optimum because
:math:`u'(0) = \infty`)

With some thought, one can show that :eq:`ee00` and :eq:`ee01` are
equivalent to

.. math::
    :label: eqeul0

    u' (c_t)
    = \max \left\{
        \beta R \,  \mathbb{E}_t [ u'(c_{t+1}) ] \,,\;  u'(a_t)
    \right\}



Optimality Results
------------------

As shown in :cite:`ma2020income`,


#. For each :math:`(a,z) \in S`, a unique optimal consumption path from :math:`(a,z)` exists

#. This path is the unique feasible path from :math:`(a,z)` satisfying the
   Euler equality :eq:`eqeul0` and the transversality condition

.. math::
    :label: eqtv

    \lim_{t \to \infty} \beta^t \, \mathbb{E} \, [ u'(c_t) a_{t+1} ] = 0


Moreover, there exists an *optimal consumption function*
:math:`\sigma^* \colon \mathsf S \to \mathbb R_+` such that the path 
from :math:`(a,z)` generated by

.. math::

    (a_0, z_0) = (a, z),
    \quad
    c_t = \sigma^*(a_t, Z_t)
    \quad \text{and} \quad
    a_{t+1} = R (a_t - c_t) + Y_{t+1} 

satisfies both :eq:`eqeul0` and :eq:`eqtv`, and hence is the unique optimal
path from :math:`(a,z)`.

Thus, to solve the optimization problem, we need to compute the policy :math:`\sigma^*`.


.. _ifp_computation:

Computation
===========

.. index::
    single: Optimal Savings; Computation

There are two standard ways to solve for :math:`\sigma^*`

#. Time iteration (TI) using the Euler equality

#. Value function iteration (VFI)

Our investigation of the cake eating problem and stochastic optimal growth
model suggest that time iteration will be faster and more accurate.

This is the approach that we apply below.

Time Iteration
--------------

We can rewrite :eq:`eqeul0` to make it a statement about functions rather than
random variables.

In particular, consider the functional equation

.. math::
    :label: eqeul1

    (u' \circ \sigma)  (a, z)
    = \max \left\{
    \beta R \, \mathbb E_z (u' \circ \sigma)  
        [R (a - \sigma(a, z)) + \hat Y, \, \hat Z]
    \, , \;
         u'(a)
         \right\}

where 

* :math:`(u' \circ c)(s) := u'(c(s))`.
* :math:`\mathbb E_z` conditions on current state :math:`z` and :math:`\hat X`
  indicates next period value of random variable :math:`X` and
* :math:`\sigma` is the unknown function.

Let :math:`\mathscr{C}` be the set of
candidate consumption functions :math:`\sigma \colon \mathsf S \to \mathbb R`
such that

* each :math:`\sigma \in \mathscr{C}` is continuous and (weakly) increasing
* :math:`0 < \sigma(a,z) \leq a` for all :math:`(a,z) \in \mathsf S`

In addition, let :math:`K \colon \mathscr{C} \to \mathscr{C}` be defined as
follows.

For given :math:`\sigma \in \mathscr{C}`, the value :math:`K \sigma (a,z)` is the unique :math:`t \in J(a,z)` that solves

.. math::
    :label: eqsifc

    u'(t)
    = \max \left\{
               \beta R \, \mathbb E_z (u' \circ \sigma) \, 
               [R (a - t) + \hat Y, \, \hat Z]
               \, , \;
               u'(a)
         \right\}

where

.. math::
    :label: eqbos

    J(a,z) := \{t \in \mathbb{R} \,:\, \min Z \leq t \leq Ra+ z + b\}


We refer to :math:`K` as Coleman's policy function operator :cite:`Coleman1990`.

The operator :math:`K` is constructed so that fixed points of :math:`K`
coincide with solutions to the functional equation :eq:`eqeul1`.

It is shown in :cite:`ma2020income` that the unique optimal policy can be
computed by picking any :math:`\sigma \in \mathscr{C}` and iterating with the
operator :math:`K` defined in :eq:`eqsifc`.

Some Technical Details
----------------------

The proof of the last statement is somewhat technical but here is a quick
summary:

It is shown in :cite:`ma2020income` that :math:`K` is a contraction mapping on
:math:`\mathscr{C}` under the metric

.. math::

    \rho(c, d) := \| \, u' \circ \sigma_1 - u' \circ \sigma_2 \, \|
        := \sup_{s \in S} | \, u'(\sigma_1(s))  - u'(\sigma_2(s)) \, |
     \qquad \quad (\sigma_1, \sigma_2 \in \mathscr{C})

It is also shown that the metric :math:`\rho` is complete on :math:`\mathscr{C}`.

In consequence, :math:`K` has a unique fixed point :math:`\sigma^* \in \mathscr{C}` and :math:`K^n c \to \sigma^*` as :math:`n \to \infty` for any :math:`\sigma \in \mathscr{C}`.

By the definition of :math:`K`, the fixed points of :math:`K` in :math:`\mathscr{C}` coincide with the solutions to :eq:`eqeul1` in :math:`\mathscr{C}`.

As a consequence, the path :math:`\{c_t\}` generated from :math:`(a_0,z_0) \in
S` using policy function :math:`\sigma^*` is the unique optimal path from
:math:`(a_0,z_0) \in S`.


Implementation
==============

.. index::
    single: Optimal Savings; Programming Implementation

We use the CRRA utility specification

.. math::
    u(c) = \frac{c^{1 - γ} - 1} {1 - γ}

The exogeneous state process :math:`\{Z_t\}` defaults to a two-state Markov chain
with state space :math:`\{0, 1\}` and transition matrix :math:`P`.

Here we build a class called ``IFP`` that stores the model primitives.

.. code-block:: python3

    ifp_data = [
        ('R', float64),              # Interest rate 1 + r
        ('β', float64),              # Discount factor
        ('γ', float64),              # Preference parameter
        ('P', float64[:, :]),        # Markov matrix for binary Z_t 
        ('y', float64[:]),           # Income is Y_t = y[Z_t]
        ('asset_grid', float64[:])   # Grid (array)
    ]

    @jitclass(ifp_data)
    class IFP:

        def __init__(self,
                     r=0.01,             
                     β=0.96,            
                     γ=1.5,            
                     P=((0.6, 0.4),
                        (0.05, 0.95)),
                     y=(0.0, 2.0),
                     grid_max=16,
                     grid_size=50):

            self.R = 1 + r
            self.β, self.γ = β, γ
            self.P, self.y = np.array(P), np.array(y)
            self.asset_grid = np.linspace(0, grid_max, grid_size)

        def u_prime(self, c):
            return c**(-self.γ)


.. code-block:: python3

    @njit
    def euler_diff(c, a, z, σ_vals, ifp):
        """
        The difference of the left-hand side and the right-hand side
        of the Euler Equation, given current policy σ.

            * c is the consumption choice
            * (a, z) is the state, with z in {0, 1}
            * σ_vals is a policy represented as a matrix.
            * ifp is an instance of IFP

        """

        # Simplify names
        R, P, y, β, γ  = ifp.R, ifp.P, ifp.y, ifp.β, ifp.γ
        asset_grid, u_prime = ifp.asset_grid, ifp.u_prime
        n = len(P)

        # Convert policy into a function by linear interpolation
        def σ(a, z):
            return interp(asset_grid, σ_vals[:, z], a)

        # Calculate the expectation conditional on current z
        expect = 0.0
        for z_hat in range(n):
            expect += u_prime(σ(R * (a - c) + y[z_hat], z_hat)) * P[z, z_hat]

        return u_prime(c) - max(β * R * expect, u_prime(a))


    @njit
    def K(σ, ifp):
        """
        The operator K.

        """
        σ_new = np.empty_like(σ)
        for i, a in enumerate(ifp.asset_grid):
            for z in (0, 1):
                result = brentq(euler_diff, 1e-8, a, args=(a, z, σ, ifp))
                σ_new[i, z] = result.root

        return σ_new


Note that we use linear interpolation along the asset grid to approximate the
policy function.

The following function iterates to convergence and returns the approximate
optimal policy.



.. literalinclude:: /_static/lecture_specific/coleman_policy_iter/solve_time_iter.py 


Let's solve using the default parameters of the ``IFP`` class:


.. code-block:: python3

    ifp = IFP()

    # Set up initial consumption policy of consuming all assets at all z
    z_size = len(ifp.P)
    a_grid = ifp.asset_grid
    a_size = len(a_grid)
    σ_init = np.repeat(a_grid.reshape(a_size, 1), z_size, axis=1)

    σ_star = solve_model_time_iter(ifp, σ_init)

Here's a plot of the resulting policy for each exogeneous state :math:`z`.

.. code-block:: python3

    fig, ax = plt.subplots()
    for z in range(z_size):
        label = rf'$\sigma^*(\cdot, {z})$'
        ax.plot(a_grid, σ_star[:, z], label=label)
    ax.set(xlabel='assets', ylabel='consumption')
    ax.legend()
    plt.show()

The following exercises walk you through several applications where policy functions are computed.



A Sanity Check
--------------

One way to check our results is to 

* set labor income to zero in each state and
* set the gross interest rate :math:`R` to unity.

In this case, our income fluctuation problem is just a cake eating problem.

We know that, in this case, the value function and optimal consumption policy
are given by 


.. literalinclude:: /_static/lecture_specific/cake_eating_numerical/analytical.py

Let's see if we match up:


.. code-block:: python3

    ifp_cake_eating = IFP(r=0.0, y=(0.0, 0.0))

    σ_star = solve_model_time_iter(ifp_cake_eating, σ_init)

    fig, ax = plt.subplots()
    ax.plot(a_grid, σ_star[:, 0], label='numerical')
    ax.plot(a_grid, c_star(a_grid, ifp.β, ifp.γ), '--', label='analytical')

    ax.set(xlabel='assets', ylabel='consumption')
    ax.legend()

    plt.show()


Success!



Exercises
=========


Exercise 1
----------

Next, let's consider how the interest rate affects consumption.

Reproduce the following figure, which shows (approximately) optimal consumption policies for different interest rates

.. figure:: /_static/lecture_specific/ifp/ifp_policies.png

* Other than ``r``, all parameters are at their default values.
* ``r`` steps through ``np.linspace(0, 0.04, 4)``.
* Consumption is plotted against assets for income shock fixed at the smallest value.

The figure shows that higher interest rates boost savings and hence suppress consumption.



Exercise 2
----------

Now let's consider the long run asset levels held by households.

We'll take ``r = 0.03`` and otherwise use default parameters.

The following figure is a 45 degree diagram showing the law of motion for assets when consumption is optimal

.. code-block:: python3

    m = IFP(r=0.03, grid_max=4)
    K = operator_factory(m)

    σ_star = solve_model(m, verbose=False)
    a = m.asset_grid
    R, z_vals = m.R, m.z_vals

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(a, R * a + z_vals[0] - σ_star[:, 0], label='Low income')
    ax.plot(a, R * a + z_vals[1] - σ_star[:, 1], label='High income')
    ax.plot(a, a, 'k--')
    ax.set(xlabel='Current assets',
           ylabel='Next period assets',
           xlim=(0, 4), ylim=(0, 4))
    ax.legend()
    plt.show()



The blue line and orange line represent the function

.. math::

    a' = h(a, z) := R a + z - \sigma^*(a, z)


when income :math:`z` takes its high and low values respectively.

The dashed line is the 45 degree line.

We can see from the figure that the dynamics will be stable --- assets do not
diverge.

In fact there is a unique stationary distribution of assets that we can calculate by simulation

* Can be proved via theorem 2 of :cite:`HopenhaynPrescott1992`.

* Represents the long run dispersion of assets across households when households have idiosyncratic shocks.


Ergodicity is valid here, so stationary probabilities can be calculated by averaging over a single long time series.

Hence to approximate the stationary distribution we can simulate a long time series for assets and histogram, as in the following figure

.. figure:: /_static/lecture_specific/ifp/ifp_histogram.png

Your task is to replicate the figure

* Parameters are as discussed above.

* The histogram in the figure used a single time series :math:`\{a_t\}` of length 500,000.

* Given the length of this time series, the initial condition :math:`(a_0, z_0)` will not matter.

* You might find it helpful to use the ``MarkovChain`` class from ``quantecon``.




Exercise 3
----------

Following on from exercises 1 and 2, let's look at how savings and aggregate asset holdings vary with the interest rate

* Note: :cite:`Ljungqvist2012` section 18.6 can be consulted for more background on the topic treated in this exercise.

For a given parameterization of the model, the mean of the stationary distribution can be interpreted as aggregate capital in an economy with a unit mass of *ex-ante* identical households facing idiosyncratic shocks.

Let's look at how this measure of aggregate capital varies with the interest
rate and borrowing constraint.

The next figure plots aggregate capital against the interest rate for ``b in (1, 3)``


.. figure:: /_static/lecture_specific/ifp/ifp_agg_savings.png

As is traditional, the price (interest rate) is on the vertical axis.

The horizontal axis is aggregate capital computed as the mean of the stationary distribution.

Exercise 3 is to replicate the figure, making use of code from previous exercises.

Try to explain why the measure of aggregate capital is equal to :math:`-b`
when :math:`r=0` for both cases shown here.

Solutions
=========




Exercise 1
----------

.. code-block:: python3

    r_vals = np.linspace(0, 0.04, 4)

    fig, ax = plt.subplots(figsize=(10, 8))
    for r_val in r_vals:
        cp = IFP(r=r_val)
        σ_star = solve_model(cp, verbose=False)
        ax.plot(cp.asset_grid, σ_star[:, 0], label=f'$r = {r_val:.3f}$')

    ax.set(xlabel='asset level', ylabel='consumption (low income)')
    ax.legend()
    plt.show()


Exercise 2
----------

.. code-block:: python3

    def compute_asset_series(cp, T=500000, verbose=False):
        """
        Simulates a time series of length T for assets, given optimal
        savings behavior.

        cp is an instance of IFP
        """
        Π, z_vals, R = cp.Π, cp.z_vals, cp.R  # Simplify names
        mc = MarkovChain(Π)
        σ_star = solve_model(cp, verbose=False)
        cf = lambda a, i_z: interp(cp.asset_grid, σ_star[:, i_z], a)
        a = np.zeros(T+1)
        z_seq = mc.simulate(T)
        for t in range(T):
            i_z = z_seq[t]
            a[t+1] = R * a[t] + z_vals[i_z] - cf(a[t], i_z)
        return a

    cp = IFP(r=0.03, grid_max=4)
    a = compute_asset_series(cp)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(a, bins=20, alpha=0.5, density=True)
    ax.set(xlabel='assets', xlim=(-0.05, 0.75))
    plt.show()


Exercise 3
----------

.. code-block:: python3

    M = 25
    r_vals = np.linspace(0, 0.04, M)
    fig, ax = plt.subplots(figsize=(10, 8))

    for b in (1, 3):
        asset_mean = []
        for r_val in r_vals:
            cp = IFP(r=r_val, b=b)
            mean = np.mean(compute_asset_series(cp, T=250000))
            asset_mean.append(mean)
        ax.plot(asset_mean, r_vals, label=f'$b = {b:d}$')
        print(f"Finished iteration b = {b:d}")

    ax.set(xlabel='capital', ylabel='interest rate')
    ax.grid()
    ax.legend()
    plt.show()
