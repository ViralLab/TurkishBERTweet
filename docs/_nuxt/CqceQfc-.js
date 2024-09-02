import{_ as d,o as r,c as s,p as m,e as u,a as n,f,b as e,w as a,g as h,F as k}from"./CSoyElds.js";import{_ as b}from"./eAXRnBeW.js";import{_ as g}from"./9MOn9I1i.js";import"./Dtdg3njw.js";import"./DRKT6Zaj.js";const x={},i=t=>(m("data-v-2b662a4b"),t=t(),u(),t),y={class:"header-background flex flex-col items-center justify-center space-y-8 bg-cover bg-center bg-no-repeat py-10 md:py-16 lg:py-24 text-gray-200"},v=i(()=>n("h1",{class:"text-6xl"},"Sentiment Analysis",-1)),S=i(()=>n("hr",{class:"w-1/12"},null,-1)),w=i(()=>n("p",{class:"text-2xl"},null,-1)),z=[v,S,w];function T(t,_){return r(),s("header",y,z)}const C=d(x,[["render",T],["__scopeId","data-v-2b662a4b"]]),A={class:"inner pt-10"},B=n("br",null,null,-1),M=`import torch
from peft import (
    PeftModel,
    PeftConfig,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer)
from Preprocessor import preprocess
 

peft_model = "VRLLab/TurkishBERTweet-Lora-SA"
peft_config = PeftConfig.from_pretrained(peft_model)

# loading Tokenizer
padding_side = "right"
tokenizer = AutoTokenizer.from_pretrained(
    peft_config.base_model_name_or_path, padding_side=padding_side
)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

id2label_sa = {0: "negative", 2: "positive", 1: "neutral"}
turkishBERTweet_sa = AutoModelForSequenceClassification.from_pretrained(
    peft_config.base_model_name_or_path, return_dict=True, num_labels=len(id2label_sa), id2label=id2label_sa
)
turkishBERTweet_sa = PeftModel.from_pretrained(turkishBERTweet_sa, peft_model)

sample_texts = [
    "Viral lab da insanlar hep birlikte çalışıyorlar. hepbirlikte çalışan insanlar birbirlerine yakın oluyorlar.",     
    "americanin diplatlari turkiyeye gelmesin 😤",
    "Mark Zuckerberg ve Elon Musk'un boks müsabakası süper olacak! 🥷",
    "Adam dun ne yediğini unuttu"
    ]


preprocessed_texts = [preprocess(s) for s in sample_texts]
with torch.no_grad():
    for s in preprocessed_texts:
        ids = tokenizer.encode_plus(s, return_tensors="pt")
        label_id = turkishBERTweet_sa(**ids).logits.argmax(-1).item()
        print(id2label_sa[label_id],":", s)
`,E=`
lab'ımıza "viral" adını verdik çünkü amacımız disiplinler arası sınırları aşmak ve aralarında yeni bağlantılar kurmak! <emoji> mikroskop </emoji> <hashtag> virallab </hashtag> <http> varollab.com </http>
`,R=f({__name:"SentimentMain",setup(t){return(_,l)=>{const o=b,c=h,p=g;return r(),s("main",A,[e(p,null,{default:a(()=>[n("div",null,[e(c,null,{default:a(()=>[e(o,{code:M,language:"python"})]),_:1}),e(c,null,{default:a(()=>[e(o,{code:E})]),_:1})]),B]),_:1})])}}}),$={};function P(t,_){const l=C,o=R;return r(),s(k,null,[e(l),e(o)],64)}const L=d($,[["render",P]]);export{L as default};
