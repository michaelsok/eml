.. _marginal:

**************
Marginal Plots
**************

We use the standard `iris` dataset for the marginal plots with a ``RandomForestClassifier`` fitted 
to provide basic marginal plots.

.. ipython:: python

    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier

    iris = load_iris()

    estimator = RandomForestClassifier()
    estimator.fit(iris.data, iris.target)

.. _marginal.pdp:

Partial dependence plots: ``PartialDependence``
-----------------------------------------------

No argument is needed for initialization while the ``fit`` method 
takes as argument an estimator with a ``predict_proba`` or ``predict`` method.
The ``interpret`` (and ``predict``) method takes data similar to the one used to fit the
estimator.

.. ipython:: python

    from eml.marginal import PartialDependence

    pdp = PartialDependence()
    pdp.fit(estimator)
    x, y = pdp.interpret(X=iris.data, feature=0, n_splits=20)

Here `x` is the abscissa of the partial dependence plot and `y` are the
ordonna of the partial dependence plot, `y` have the same dimension than
the estimator ``predict`` method.


For the following plots, we use the standard convention for referencing Plotly:

.. ipython:: python

    import plotly.graph_objects as go
    from plotly.offline import plot
    from plotly.subplots import make_subplots

Also the different plots will be saved in the directory ``plots``:

.. ipython:: python

    import os
    os.makedirs('plots', exist_ok=True)


Below is a simple subplot figure with plotly for multi-classification partial dependence plot:

.. ipython:: python

    fig = make_subplots(rows=2, cols=2, subplot_titles=iris.target_names)
    feature_name = iris.feature_names[0]

    for idx, (output, name) in enumerate(zip(y.T, iris.target_names)):
        row, col = (idx // 2) + 1, (idx % 2) + 1
        fig.add_trace(go.Scatter(x=x, y=output, name=name, mode='lines+markers'),
                      row=row, col=col)
        fig.update_xaxes(title=f'{feature_name} values', row=row, col=col)
        fig.update_yaxes(title='Probabilities', range=(0, 1), row=row, col=col)

    fig = fig.update_layout(title=f'{feature_name} partial dependence plot', height=600)
    plot(fig, filename='plots/pdp_basic.html', auto_open=False) # to save the plot
    fig.show()

.. raw:: html
    :file: ../../plots/pdp_basic.html

We also provide a partial dependence plot with a standard deviation interval.
First the lower and upper limits are available through the ``interpret`` method:

.. ipython:: python

    x, y, (low, up) = pdp.interpret(iris.data, feature=0, with_confidence=True, coef=1)

Then you can use plotly `fill='tonexty'` option.

.. ipython:: python

    fig = make_subplots(rows=2, cols=2, subplot_titles=iris.target_names)
    colors = (
        'rgba(31, 119, 180, {opacity})',
        'rgba(255, 127, 14, {opacity})',
        'rgba(44, 160, 44, {opacity})'
    )
    arguments = zip(y.T, low.T, up.T, iris.target_names, colors)

    for idx, (output, l, u, n, c) in enumerate(arguments):
        row, col = (idx // 2) + 1, (idx % 2) + 1
        trace_pdp = go.Scatter(x=x, y=output, name=n,
                              line_color=c.format(opacity=1), mode='lines+markers')
        trace_up = go.Scatter(x=x, y=u, name=n,
                              line_color=c.format(opacity=0), showlegend=False)
        trace_low = go.Scatter(x=x, y=l, fill='tonexty',
                               fillcolor=c.format(opacity=0.2),
                               mode='none', name='Standard Deviation')
        fig.add_trace(trace_up, row=row, col=col)
        fig.add_trace(trace_low, row=row, col=col)
        fig.add_trace(trace_pdp, row=row, col=col)
        fig.update_xaxes(title=f'{feature_name} values', row=row, col=col)
        fig.update_yaxes(title='Probabilities', range=(-.5, 1.5), row=row, col=col)
    
    fig = fig.update_layout(title=f'{feature_name} partial dependence plot', height=600)
    plot(fig, filename='plots/pdp_confidence.html', auto_open=False) # to save the plot
    fig.show()

.. raw:: html
    :file: ../../plots/pdp_confidence.html

.. _marginal.ice:

ICE Curves: ``ICE``
-------------------

Similar to ``PartialDependencePlot``, the initialization and the ``fit`` method takes
as argument an estimator with ``predict_proba`` or ``predict`` method. The ``interpret`` 
(and ``predict``) method takes data similar to the one used to fit the estimator.

.. ipython:: python

    from eml.marginal import ICE

    ice = ICE()
    ice.fit(estimator)
    x, y = ice.interpret(X=iris.data, feature=2, n_splits=20)

Below is a simple subplot figure with plotly for multi-classification ice curves:

.. ipython:: python

    fig = make_subplots(rows=2, cols=2, subplot_titles=iris.target_names)
    feature_name = iris.feature_names[2]

    for idx, (output, name, color) in enumerate(zip(y.T, iris.target_names, colors)):
        row, col = (idx // 2) + 1, (idx % 2) + 1
        lc = color.format(opacity=.2)
        partial_line = output.mean(axis=0)
        for line in output:
            l = go.Scatter(x=x, y=line, mode='lines', line_color=lc, showlegend=False)
            fig.add_trace(l, row=row, col=col)
        pdp_trace = go.Scatter(x=x, y=partial_line, line_color=color.format(opacity=1),
                                mode='lines+markers', name=f'PDP for {name}')
        fig.add_trace(pdp_trace, row=row, col=col)
        fig.update_xaxes(title=f'{feature_name} values', row=row, col=col)
        fig.update_yaxes(title='Probabilities', range=(0, 1), row=row, col=col)

    fig = fig.update_layout(title=f'{feature_name} ICE curves', height=600)
    plot(fig, filename='plots/ice_basic.html', auto_open=False) # to save the plot
    fig.show()

.. raw:: html
    :file: ../../plots/ice_basic.html
