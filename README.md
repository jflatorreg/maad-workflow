# MAAD Work Flux
MAAD's Work Flux are the steps to follow (scripts) to perform automatic classification of sounds of interest. These steps are based on the ***MAAD*** method.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install MAAD library.

```bash
pip install upgrade -i https://test.pypi.org/simple/scikit-maad
```

## Usage

```python
from maad.rois import find_rois_cwt
from maad import sound

s, fs = sound.load('./templates/BETA-_20161006_002000_section.wav')
rois = find_rois_cwt(s, fs, 
                     flims = (1000,4000), 
                     tlen = 0.3, 
                     th = 0.0001, 
                     display=True, figsize=(13,6))
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
