# FASTAUG

`fastaug` is an NLP library for data augmentation with high speed.

## Installation

```bash
pip install git+https://github.com/dugu9sword/fastaug.git
```


## Usage

More examples can be found in `examples/*.ipynb`.

**Code**

```python
from fastaug import Augmentor
from fastaug import WordMorphSub, WordNetSub, WordRandomMask
from fastaug import CharRandomSwap, CharRandomDelete
aug = Augmentor(
    [
        # play -> playing, things -> thing
        WordMorphSub(0.1),      
        # good -> great
        WordNetSub(0.1),        
        # hello -> _
        WordRandomMask(0.1, "[MASK]"),  
        # happy -> hpapy  
        CharRandomSwap(0.1),
        # trick -> trck    
        CharRandomDelete(0.1)  
    ]
)
aug.augment("Five score years ago, a great American, in whose symbolic shadow we stand today, signed the Emancipation Proclamation.")
```

**Result**

```
['Five tally years ago, a big American, into whose sybmolic shdow we stand today, signed the Emancipation [MASK]',
 'Five seduce yeas ago, a great American, in whose symbolic shadow we stand today, [MASK] the Emancipation Proclamation.',
 'Five score class ago, a great American, in whsoe symbolic shadow we stand today, signing te Emancipation [MASK]']
```