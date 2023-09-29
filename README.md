#### Table of contents
1. [Introduction](#introduction)
2. [Main results](#results)
3. [Using TurkishBERTweet with `transformers`](#transformers)
    - [Pre-trained models](#models2)
    - [Example usage](#usage2)
    - [Normalize raw input Tweets](#preprocess)
4. [Citation](#citation)
# <a name="introduction"></a> TurkishBERTweet in the shadow of Large Language Models

<!-- ## Results
|   Dataset  |  Roberta   |            |            |            |
|------------|------------|------------|------------|------------|
| 1          |            |            |            |            |
| 2          |            |            |            |            |
| 3          |            |            |            |            | -->

<!-- https://huggingface.co/VRLLab/TurkishBERTweet -->

# <a name="usage2"></a> Example usage

```python
import torch
from transformers import AutoTokenizer
from Preprocessor import preprocess

tokenizer = AutoTokenizer.from_pretrained("VRLLab/TurkishBERTweet")
turkishBERTweet = AutoModel.from_pretrained("VRLLab/TurkishBERTweet")

text = """Lab'ımıza "viral" adını verdik çünkü amacımız disiplinler arası sınırları aşmak ve aralarında yeni bağlantılar kurmak! 💥🔬 #ViralLab #DisiplinlerArası #YenilikçiBağlantılar"""

preprocessed_text = preprocess(text)
input_ids = torch.tensor([tokenizer.encode(preprocessed_text)])

with torch.no_grad():
    features = turkishBERTweet(input_ids)  # Models outputs are now tuples
```



# <a name="citation"></a> Citation
```bibtex
@article{najafi2022TurkishBERTweet,
    title={TurkishBERTweet in the shadow of Large Language Models},
    author={Najafi, Ali and Varol, Onur},
    journal={arXiv preprint },
    year={2023}
}
```

## Acknowledgments
We thank [Fatih Amasyali](https://avesis.yildiz.edu.tr/amasyali) for providing access to Tweet Sentiment datasets from Kemik group.
This material is based upon work supported by the Google Cloud Research Credits program with the award GCP19980904. We also thank TUBITAK (121C220 and 222N311) for funding this project. 