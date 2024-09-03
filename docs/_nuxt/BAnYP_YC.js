import{_ as u,o as s,c as r,p,e as h,a as t,f as m,b as e,w as o,d as g,g as f,F as k}from"./NRmH45FU.js";import{_ as b}from"./R3PhC717.js";import{_ as y}from"./BZuIh1sg.js";import"./D8rPkWOK.js";import"./gFzsWSki.js";const T={},l=a=>(p("data-v-96987dbc"),a=a(),h(),a),v={class:"header-background flex flex-col items-center justify-center space-y-8 bg-cover bg-center bg-no-repeat py-10 md:py-16 lg:py-24 text-gray-200"},w=l(()=>t("h1",{class:"text-6xl"},"Hate Speech Detection",-1)),x=l(()=>t("hr",{class:"w-1/12"},null,-1)),B=l(()=>t("p",{class:"text-2xl"},null,-1)),E=[w,x,B];function R(a,c){return s(),r("header",v,E)}const z=u(T,[["render",R],["__scopeId","data-v-96987dbc"]]),S={class:"inner pt-10"},H=t("h3",null,"HateSpeech Detection with TurkishBERTweet",-1),C=t("p",null,"In this section, we will guide you through the process of using TurkishBERTweet for detecting hate speech. TurkishBERTweet is specifically designed to analyze Turkish social media content, making it a powerful tool for identifying harmful language, ensuring a safer online environment. ",-1),$=t("h3",null,"Setting Up the Environment for TurkishBERTweet",-1),V=t("br",null,null,-1),I=t("br",null,null,-1),M=t("h5",null,"1. Clone the Repository: Begin by cloning the TurkishBERTweet repository from GitHub.",-1),N=t("h5",null,"2. Navigate to the Directory: Move into the newly cloned directory.",-1),P=t("h5",null,"3. Set Up a Virtual Environment: Create a virtual environment to manage dependencies.",-1),A=t("h5",null,"4. Activate the Virtual Environment: Activate the virtual environment.",-1),F=t("h5",null,"5. Install Required Libraries: Install PyTorch and other essential libraries to run TurkishBERTweet.",-1),L=t("h5",null,"6. Below is a Python script that uses TurkishBERTweet to detect hate speech in sample texts:",-1),q=t("br",null,null,-1),D="git clone git@github.com:ViralLab/TurkishBERTweet.git",U="cd TurkishBERTweet",Y="python -m venv venv",j="source venv/bin/activate",G=`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install peft
pip install transformers
pip install urlextract`,O=`import torch
from peft import (
    PeftModel,
    PeftConfig,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer)
from Preprocessor import preprocess
 

peft_model = "VRLLab/TurkishBERTweet-Lora-HS"
peft_config = PeftConfig.from_pretrained(peft_model)

# loading Tokenizer
padding_side = "right"
tokenizer = AutoTokenizer.from_pretrained(
    peft_config.base_model_name_or_path, padding_side=padding_side
)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

id2label_hs = {0: "No", 1: "Yes"}
turkishBERTweet_hs = AutoModelForSequenceClassification.from_pretrained(
    peft_config.base_model_name_or_path, return_dict=True, num_labels=len(id2label_hs), id2label=id2label_hs
)
turkishBERTweet_hs = PeftModel.from_pretrained(turkishBERTweet_hs, peft_model)


sample_texts = [
    "Viral lab da insanlar hep birlikte çalışıyorlar. hepbirlikte çalışan insanlar birbirlerine yakın oluyorlar.",     
    "kasmayin artik ya kac kere tanik olduk bu azgin tehlikeli “multecilerin” yaptiklarina? bir afgan taragindan kafasi tasla ezilip tecavuz edilen kiza da git boyle cihangir solculugu yap yerse?",
    ]


preprocessed_texts = [preprocess(s) for s in sample_texts]
with torch.no_grad():
    for s in preprocessed_texts:
        ids = tokenizer.encode_plus(s, return_tensors="pt")
        label_id = turkishBERTweet_hs(**ids).logits.argmax(-1).item()
        print(id2label_hs[label_id],":", s)
`,J=`No : viral lab da insanlar hep birlikte çalışıyorlar. hepbirlikte çalışan insanlar birbirlerine yakın oluyorlar.
Yes : kasmayin artik ya kac kere tanik olduk bu azgin tehlikeli “multecilerin” yaptiklarina? bir afgan taragindan kafasi tasla ezilip tecavuz edilen kiza da git boyle cihangir solculugu yap yerse?
`,K=m({__name:"HateSpeechMain",setup(a){return(c,_)=>{const n=b,i=f,d=y;return s(),r("main",S,[e(d,null,{default:o(()=>[t("div",null,[H,C,$,g(" To begin using the TurkishBERTweet model for sentiment analysis, you'll first need to set up your development environment. This involves cloning the TurkishBERTweet repository, creating a virtual environment, and installing the necessary libraries. Follow these steps to get started: "),V,I,M,e(i,null,{default:o(()=>[e(n,{code:D,language:"bash"})]),_:1}),N,e(i,null,{default:o(()=>[e(n,{code:U,language:"bash"})]),_:1}),P,e(i,null,{default:o(()=>[e(n,{code:Y,language:"bash"})]),_:1}),A,e(i,null,{default:o(()=>[e(n,{code:j,language:"bash"})]),_:1}),F,e(i,null,{default:o(()=>[e(n,{code:G,language:"bash"})]),_:1}),L,e(i,null,{default:o(()=>[e(n,{code:O,language:"python"})]),_:1}),e(i,null,{default:o(()=>[e(n,{code:J})]),_:1})]),q]),_:1})])}}}),Q={};function W(a,c){const _=z,n=K;return s(),r(k,null,[e(_),e(n)],64)}const oe=u(Q,[["render",W]]);export{oe as default};
