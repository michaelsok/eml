.. _importances:

*******************
Feature Importances
*******************

We use the standard `iris` dataset for the feature importances with a ``RandomForestClassifier`` fitted 
to provide examples of plots on Mean Decrease in Impurity (MDI) importances.

.. ipython:: python

    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier

    iris = load_iris()

    estimator = RandomForestClassifier()
    estimator.fit(iris.data, iris.target)

.. _importances.mdi:

Mean Decrease in Impurity: ``MeanDecreaseImpurity``
---------------------------------------------------

During initialization you can choose to set the ``use_precompute`` option to True which allows 
to use precomputed feature importances from the different API (scikit-learn for instance).
The ``fit`` method takes a fitted decision tree or forest as argument with ``predict`` or 
``predict_proba`` as method.
The ``interpret`` (and ``predict``) method takes data similar to the one used to fit the estimator.

.. ipython:: python

    from eml.importances import MeanDecreaseImpurity

    mdi = MeanDecreaseImpurity(use_precompute=False)
    mdi.fit(estimator)
    importances = mdi.interpret(X=iris.data)

The ``importances`` are the feature importances in same order than the features in the input data.

For the following plots, we use the standard convention for referencing Plotly:

.. ipython:: python

    import plotly.graph_objects as go
    from plotly.offline import plot

Also the different plots will be saved in the directory ``plots``:

.. ipython:: python

    import os
    os.makedirs('plots', exist_ok=True)

Below is a simple subplot figure with plotly for MDI feature importances:

.. ipython:: python

    ranking = np.argsort(importances)
    ranked_importances = importances[ranking]
    ranked_features = np.array(iris.feature_names)[ranking]
    data = [go.Bar(x=ranked_importances, y=ranked_features, orientation='h')]
    layout = go.Layout(title='MDI importances')
    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename='plots/mdi_basic.html', auto_open=False) # to save the plot
    fig.show()

.. raw:: html
    :file: ../../plots/mdi_basic.html


Permutation Importances: ``PermutationImportances``
---------------------------------------------------

The Permutation importances also known as Mean Decrease in Accuracy importances 
(a more correct name should be Mean Decrease in Score) uses the decrease in a scoring 
function after permutation in one feature to see how the scoring behaves.

Thus, during initialization you have to choose a scoring function which takes as input the ``predict_proba`` 
or ``predict`` method of an estimator given during the ``fit`` method. If no ``scoring`` is passed at 
initialization, the class expect the estimator to have a ``score`` method.
The ``interpret`` (and ``predict``) method takes data similar to the one used to fit the estimator and 
a ground truth. 
There are also other arguments such as the number of iterations used to estimate the importances, and other options 
such as the choice to get the permutaion importances computed for each permutation:

.. ipython:: python

    from eml.importances import PermutationImportances

    mda = PermutationImportances(scoring=None)
    mda.fit(estimator)
    importances = mda.interpret(X=iris.data, y=iris.target, n_iter=10)

Below is a simple subplot figure with plotly for permutation importances:

.. ipython:: python

    ranking = np.argsort(importances)
    ranked_importances = importances[ranking]
    ranked_features = np.array(iris.feature_names)[ranking]
    data = [go.Bar(x=ranked_importances, y=ranked_features, orientation='h')]
    layout = go.Layout(title='Permutation importances with 10 iterations')
    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename='plots/mda_basic.html', auto_open=False) # to save the plot
    fig.show()

.. raw:: html
    :file: ../../plots/mda_basic.html

