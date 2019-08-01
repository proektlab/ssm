.. SSM documentation master file, created by
   sphinx-quickstart on Wed Jul 31 14:37:33 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SSM: Learning and Inference in State Space Models
=================================================

State space models are probabilistic models for sequential data.  Hidden Markov models (HMM’s) are the canonical example: the observed data is modeled as a function of an underlying latent discrete state that randomly switches from one time-step to the next according to a transition matrix. Modern state space models have expanded on this basic formula in a variety of ways: by considering other types of latent states, more complex transitions and dynamics, and more sophisticated mappings between the latent state and the observed data.  Importantly, recent advances in artificial neural networks and deep learning have enabled us to capture highly nonlinear mappings and transition rules, which we can think of as learning an underlying data manifold and the dynamics on that manifold.  “Fitting” these models amounts to a) learning the parameters of the transitions and mappings, and b) inferring the latent states underlying the observed data.  These tools have countless applications in biological contexts, where many datasets come in the form of complex spatial and/or temporal sequences; e.g. genetic sequencing data, measurements of neural activity, videos of animal behavior, and many more.

SSM instantiates these ideas in code according to a few key design principles:

1. _There is a logical distinction between the model and the inference algorithm. SSM should make it easy for scientists to iterate on the model (where they are domain experts) without having to worry about the inference algorithm._

A model encapsulates parameters and determines the probability of the latent states and observed data; an inference algorithm outputs a posterior distribution of latent states given the observed data and the model parameters.  SSM makes this separation clear in code via different objects for models and posteriors.

Since many inference algorithms only require first or second derivatives of the model probability, the model is designed to support automatic differentiation.  When a user needs to tweak a model for his or her particular scientific problem, they only need to specify how the probability is calculated.  Inference algorithms are generally harder to write, so the built-in algorithms should nearly always suffice.

2. _SSM should scale to very large datasets and capitalize on available hardware resources._

We would like SSM to run seamlessly on either a desktop or a GPU/TPU cluster without much intervention from the user.  This is essential to science, since many users run exploratory analyses locally before running complete analyses of a full dataset, which typically requires a compute cluster. To accommodate these two use-cases, our algorithms must be parallelizable and our implementation must rely on computing libraries with broad hardware support.

3. _Classical models should “just work,” standard inference algorithms should be fast, and worked examples should illustrate best practices on real datasets._

The vast majority of users do not need complicated tools; they need classical tools that work out of the box and don’t require a supercomputer.  SSM puts an emphasis on sensible defaults and intelligent initialization schemes.  Moreover, SSM has a fast library of low-level message passing routines, which form the core of many inference algorithms and make fitting standard models fast and efficient. Finally, SSM should have an array of worked examples, with real data, that illustrate how to use these tools responsibly and perform the proper checks and controls.


.. toctree::
   :maxdepth: 2
   :caption: Contents:


.. automodule:: ssm.hmm
   :members:

.. automodule:: ssm.lds
   :members:

.. automodule:: ssm.transitions
   :members:

.. automodule:: ssm.observations
   :members:

.. automodule:: ssm.emissions
   :members:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
