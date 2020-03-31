# ST-OPTICS

**Simple and effective tool for spatial-temporal clustering**

*st_optics* is an open-source software package for the spatial-temporal clustering of movement data:

- Implemnted using `numpy` and `sklearn`
- Enables to also scale to memory - with splitting the data into frames
- __Usage:__ can view a demo of common features in this
[this Jupyter Notebook](/demo/demo.ipynb).

## Installation
The easiest way to install *st_optics* is by using `pip` :

    pip install st_optics

## How to use

```python
from st_optics import ST_OPTICS

st_optics = ST_OPTICS(xi = 0.4, eps2 = 10, min_samples = 5)
st_optics.fit(data)
```

## Description

A package to perform the ST OPTICS clustering. For more details please see the following papers:

* Ankerst, M., Breunig, M. M., Kriegel, H. P., & Sander, J. (1999). OPTICS: ordering points to identify the clustering structure. ACM Sigmod record, 28(2), 49-60.

## License
This package was developed by Eren Cakmak from the [Data Analysis and Visualization Group](https://www.vis.uni-konstanz.de/) and the [Department of Collective Behaviour](http://collectivebehaviour.com) at the University Konstanz funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's ExcellenceStrategy – EXC 2117 – 422037984“

Released under MIT License. See the [LICENSE](LICENSE) file for details.
