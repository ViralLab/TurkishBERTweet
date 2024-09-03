import{_ as u,o as a,c as r,p,e as h,a as t,f as m,b as e,w as o,d as g,g as f,F as b}from"./NRmH45FU.js";import{_ as k}from"./R3PhC717.js";import{_ as y}from"./BZuIh1sg.js";import"./D8rPkWOK.js";import"./gFzsWSki.js";const v={},l=s=>(p("data-v-2b662a4b"),s=s(),h(),s),T={class:"header-background flex flex-col items-center justify-center space-y-8 bg-cover bg-center bg-no-repeat py-10 md:py-16 lg:py-24 text-gray-200"},w=l(()=>t("h1",{class:"text-6xl"},"Sentiment Analysis",-1)),x=l(()=>t("hr",{class:"w-1/12"},null,-1)),E=l(()=>t("p",{class:"text-2xl"},null,-1)),B=[w,x,E];function R(s,_){return a(),r("header",T,B)}const S=u(v,[["render",R],["__scopeId","data-v-2b662a4b"]]),z={class:"inner pt-10"},A=t("h3",null,"Sentiment Analysis with TurkishBERTweet",-1),C=t("p",null,"In this section, we will guide you through the process of using TurkishBERTweet for sentiment analysis. TurkishBERTweet is a specialized model designed to analyze Turkish social media text, enabling you to determine the sentiment behind tweets, posts, and comments with high accuracy. ",-1),M=t("h3",null,"Setting Up the Environment for TurkishBERTweet",-1),$=t("br",null,null,-1),V=t("br",null,null,-1),I=t("h5",null,"1. Clone the Repository: Begin by cloning the TurkishBERTweet repository from GitHub.",-1),P=t("h5",null,"2. Navigate to the Directory: Move into the newly cloned directory.",-1),j=t("h5",null,"3. Set Up a Virtual Environment: Create a virtual environment to manage dependencies.",-1),F=t("h5",null,"4. Activate the Virtual Environment: Activate the virtual environment.",-1),H=t("h5",null,"5. Install Required Libraries: Install PyTorch and other essential libraries to run TurkishBERTweet.",-1),L=t("h5",null,"6. Run the following script to get the sentiments of the sample texts:",-1),N=t("br",null,null,-1),q="git clone git@github.com:ViralLab/TurkishBERTweet.git",U="cd TurkishBERTweet",D="python -m venv venv",G="source venv/bin/activate",O=`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install peft
pip install transformers
pip install urlextract`,Z=`import torch
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
`,J=`positive : viral lab da insanlar hep birlikte çalışıyorlar. hepbirlikte çalışan insanlar birbirlerine yakın oluyorlar.
negative : americanin diplatlari turkiyeye gelmesin <emoji> burundan_buharla_yüzleşmek </emoji>
positive : mark zuckerberg ve elon musk'un boks müsabakası süper olacak! <emoji> kadın_muhafız_koyu_ten_tonu </emoji>
neutral : adam dun ne yediğini unuttu`,K=m({__name:"SentimentMain",setup(s){return(_,c)=>{const n=k,i=f,d=y;return a(),r("main",z,[e(d,null,{default:o(()=>[t("div",null,[A,C,M,g(" To begin using the TurkishBERTweet model for sentiment analysis, you'll first need to set up your development environment. This involves cloning the TurkishBERTweet repository, creating a virtual environment, and installing the necessary libraries. Follow these steps to get started: "),$,V,I,e(i,null,{default:o(()=>[e(n,{code:q,language:"bash"})]),_:1}),P,e(i,null,{default:o(()=>[e(n,{code:U,language:"bash"})]),_:1}),j,e(i,null,{default:o(()=>[e(n,{code:D,language:"bash"})]),_:1}),F,e(i,null,{default:o(()=>[e(n,{code:G,language:"bash"})]),_:1}),H,e(i,null,{default:o(()=>[e(n,{code:O,language:"bash"})]),_:1}),L,e(i,null,{default:o(()=>[e(n,{code:Z,language:"python"})]),_:1}),e(i,null,{default:o(()=>[e(n,{code:J})]),_:1})]),N]),_:1})])}}}),Q={};function W(s,_){const c=S,n=K;return a(),r(b,null,[e(c),e(n)],64)}const oe=u(Q,[["render",W]]);export{oe as default};
