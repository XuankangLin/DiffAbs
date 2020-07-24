from pathlib import Path

from diffabs.abs import AbsDom


class Dom(AbsDom):
    name = Path(__file__).with_suffix('').name  # use file name (without extension) as domain name

    def __getattr__(self, name: str) -> object:
        assert name in globals()
        return eval(name)
    pass


d = Dom()
print(d)