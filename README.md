# The Export Predictor

The **Export Predictor** is a machine learning tool to identify viable export industries and latent comparative
advantage. It was designed primarily to help governments in developing countries prioritise their
industrial policy. The accompanying [website](https://peterjrichens.github.io/ExportPredictor) allows users to
visualise the predictions, as well as the country proximity networks used in the model.

This repository contains Python scripts to download international trade data via the
[UN Comtrade API](https://comtrade.un.org/data/doc/api), update a PostgreSQL database, extract features, train
and validate the predictive model. The results are visualised
using [D3](https://github.com/d3/d3) and [D3plus](https://github.com/alexandersimoes/d3plus). [Read more](https://peterjrichens.github.io/ExportPredictor/about.html).

The Export Predictor was built by [Peter Richens](https://sg.linkedin.com/in/peter-richens) as a project for
the [General Assembly](https://generalassemb.ly/education/data-science) Data Science course.
