# DiffAbs

DiffAbs is a PyTorch implementation of multiple abstract domains that
can be used in certifying or reasoning neural networks. Implemented
purely using PyTorch, it is differentiable and supports GPU by
default, thus amenable for safety/robustness driven training on
abstract domains.

Currently, the following abstract domains are implemented:

* Vanilla interval domain;
* DeepPoly domain (<https://dl.acm.org/doi/10.1145/3290354>);


## Domain notes

DeepPoly ReLU heuristics:

* A variant of the original DeepPoly domain is implemented where the
  ReLU approximation is not heuristically choosing between two choices
  (either picking `y = x` or `y = 0` as the new upper bound). Right
  now it is fixed to choosing `y = 0`, because there was Galois
  connection violation observed if this heuristic is
  enabled. Basically, it is observed in experiment that a smaller
  abstraction may unexpectedly incur larger safety distance than its
  containing larger abstraction.


## Supported systems

Although it is currently tested on Mac OS X 10.15 and Ubuntu 16.04
with Python 3.7 and PyTorch 1.5, it should generalize to other
platforms and older PyTorch (perhaps ≥ v1.0) smoothly.

However, Python ≤ 3.6 may be incompatible. Because type annotations
are specified everywhere and the type annotation of self class is only
supported by `__future__.annotations` in Python 3.7. If using Python
3.6, this needs to use 'type string' instead.


## Installation

In your virtual environment, either install directly from this repository by
```
git clone git@github.com:XuankangLin/DiffAbs.git
cd DiffAbs
pip install -e .
```
or directly install from PyPI:
```
pip install diffabs
```

## Testing

Test cases for individual abstract domains are under the `test/`
directory and can be run using command
```
pytest
```


## License

The project is available open source under the terms of [MIT
License](https://opensource.org/licenses/MIT).
