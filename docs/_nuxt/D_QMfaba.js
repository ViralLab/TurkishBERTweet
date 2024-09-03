import{_ as d,o as l,c as r,p,e as m,a as t,f,b as e,w as n,d as g,g as k,F as b}from"./BIzfX50i.js";import{_ as y}from"./P48xGgOh.js";import{_ as v}from"./BGjkTpzP.js";import"./BFS2ZWOy.js";import"./CSQE_OY1.js";const T={},c=a=>(p("data-v-96987dbc"),a=a(),m(),a),w={class:"header-background flex flex-col items-center justify-center space-y-8 bg-cover bg-center bg-no-repeat py-10 md:py-16 lg:py-24 text-gray-200"},x=c(()=>t("h1",{class:"text-6xl"},"Hate Speech Detection",-1)),B=c(()=>t("hr",{class:"w-1/12"},null,-1)),E=c(()=>t("p",{class:"text-2xl"},null,-1)),R=[x,B,E];function z(a,s){return l(),r("header",w,R)}const S=d(T,[["render",z],["__scopeId","data-v-96987dbc"]]),H={class:"inner pt-10"},C=t("h3",null,"HateSpeech Detection with TurkishBERTweet",-1),$=t("p",null,"In this section, we will guide you through the process of using TurkishBERTweet for detecting hate speech. TurkishBERTweet is specifically designed to analyze Turkish social media content, making it a powerful tool for identifying harmful language, ensuring a safer online environment. ",-1),M=t("h3",null,"Setting Up the Environment for TurkishBERTweet",-1),V=t("br",null,null,-1),I=t("br",null,null,-1),N=t("h5",null,"1. Clone the Repository: Begin by cloning the TurkishBERTweet repository from GitHub.",-1),P=t("h5",null,"2. Navigate to the Directory: Move into the newly cloned directory.",-1),A=t("h5",null,"3. Set Up a Virtual Environment: Create a virtual environment to manage dependencies.",-1),L=t("h5",null,"4. Activate the Virtual Environment: Activate the virtual environment.",-1),F=t("h5",null,"5. Install Required Libraries: Install PyTorch and other essential libraries to run TurkishBERTweet.",-1),q=t("h5",null,"6. Below is a Python script that uses TurkishBERTweet to detect hate speech in sample texts:",-1),D=t("br",null,null,-1),U="git clone git@github.com:ViralLab/TurkishBERTweet.git",Y="cd TurkishBERTweet",j="python -m venv venv",G=`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
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
`,W=`No : viral lab da insanlar hep birlikte çalışıyorlar. hepbirlikte çalışan insanlar birbirlerine yakın oluyorlar.
Yes : kasmayin artik ya kac kere tanik olduk bu azgin tehlikeli “multecilerin” yaptiklarina? bir afgan taragindan kafasi tasla ezilip tecavuz edilen kiza da git boyle cihangir solculugu yap yerse?
`,J="python run_hatespeech.py",K=f({__name:"HateSpeechMain",setup(a){const s=String.raw`#Linux/Mac users
source venv/bin/activate
# Windows users
.\venv\Scripts\activate
`;return(_,u)=>{const i=y,o=k,h=v;return l(),r("main",H,[e(h,null,{default:n(()=>[t("div",null,[C,$,M,g(" To begin using the TurkishBERTweet model for sentiment analysis, you'll first need to set up your development environment. This involves cloning the TurkishBERTweet repository, creating a virtual environment, and installing the necessary libraries. Follow these steps to get started: "),V,I,N,e(o,null,{default:n(()=>[e(i,{code:U,language:"bash",filename:"clone.sh"})]),_:1}),P,e(o,null,{default:n(()=>[e(i,{code:Y,language:"bash",filename:"cd.sh"})]),_:1}),A,e(o,null,{default:n(()=>[e(i,{code:j,language:"bash",filename:"venv.sh"})]),_:1}),L,e(o,null,{default:n(()=>[e(i,{code:s,language:"bash",filename:"activate.sh"})]),_:1}),F,e(o,null,{default:n(()=>[e(i,{code:G,language:"text",filename:"install.sh"})]),_:1}),q,e(o,null,{default:n(()=>[e(i,{code:J,language:"bash",filename:"run_finetune.sh"})]),_:1}),e(o,null,{default:n(()=>[e(i,{code:O,language:"python",filename:"run_hatespeech.py"})]),_:1}),e(o,null,{default:n(()=>[e(i,{code:W,filename:"output.txt"})]),_:1})]),D]),_:1})])}}}),Q={};function X(a,s){const _=S,u=K;return l(),r(b,null,[e(_),e(u)],64)}const oe=d(Q,[["render",X]]);export{oe as default};
