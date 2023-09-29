#### Table of contents
1. [Introduction](#introduction)
2. [Main results](#results)
3. [Using TurkishBERTweet with `transformers`](#transformers)
    - [Models](#trainedModels)
    - [Example usage](#usage2)
        - [Twitter Preprocessor](#preprocess)
        - [Feature Extraction](#feature_extraction)

4. [Citation](#citation)
# <a name="introduction"></a> TurkishBERTweet in the shadow of Large Language Models


# <a name="results"></a> Main Results
![alt text](main_results.png "Title")



<!-- https://huggingface.co/VRLLab/TurkishBERTweet -->
# <a name="trainedModels"></a> Models
Model | #params | Arch. | Max length | Pre-training data
---|---|---|---|---
`VRLLab/TurkishBERTweet` | 163M | base | 128 | 894M Turkish Tweets (uncased)

# <a name="usage2"></a> Example usage


## <a name="preprocess"></a> Twitter Preprocessor
```python
from Preprocessor import preprocess

text = """Lab'ımıza "viral" adını verdik çünkü amacımız disiplinler arası sınırları aşmak ve aralarında yeni bağlantılar kurmak! 🔬 #ViralLab
https://varollab.com/"""

preprocessed_text = preprocess(text)
print(preprocessed_text)
```
Output:
```output
lab'ımıza "viral" adını verdik çünkü amacımız disiplinler arası sınırları aşmak ve aralarında yeni bağlantılar kurmak! <emoji> mikroskop </emoji> <hashtag> virallab </hashtag> <http> varollab.com </http>
```


## <a name="feature_extraction"></a> Feature Extraction

```python
import torch
from transformers import AutoTokenizer, AutoModel
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