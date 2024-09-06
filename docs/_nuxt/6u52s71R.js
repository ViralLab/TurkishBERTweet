import{_ as h,a as d}from"./DXHoEFTA.js";import{_ as p}from"./BtIylxeA.js";import{g as m,o as r,c as u,b as e,w as n,a as t,d as f,h as g,_ as k,F as b}from"./DAmhm2ud.js";import"./C1v8UIkE.js";import"./Dg_zS6C8.js";const y={class:"inner pt-10"},T=t("h3",null,"HateSpeech Detection with TurkishBERTweet",-1),v=t("p",null,"In this section, we will guide you through the process of using TurkishBERTweet for detecting hate speech. TurkishBERTweet is specifically designed to analyze Turkish social media content, making it a powerful tool for identifying harmful language, ensuring a safer online environment. ",-1),w=t("h3",null,"Setting Up the Environment for TurkishBERTweet",-1),x=t("br",null,null,-1),B=t("br",null,null,-1),E=t("h5",null,"1. Clone the Repository: Begin by cloning the TurkishBERTweet repository from GitHub.",-1),R=t("h5",null,"2. Navigate to the Directory: Move into the newly cloned directory.",-1),z=t("h5",null,"3. Set Up a Virtual Environment: Create a virtual environment to manage dependencies.",-1),S=t("h5",null,"4. Activate the Virtual Environment: Activate the virtual environment.",-1),C=t("h5",null,"5. Install Required Libraries: Install PyTorch and other essential libraries to run TurkishBERTweet.",-1),H=t("h5",null,"6. Below is a Python script that uses TurkishBERTweet to detect hate speech in sample texts:",-1),M=t("br",null,null,-1),V="git clone git@github.com:ViralLab/TurkishBERTweet.git",N="cd TurkishBERTweet",P="python -m venv venv",A=`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install peft
pip install transformers
pip install urlextract`,L=`import torch
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
`,F=`No : viral lab da insanlar hep birlikte çalışıyorlar. hepbirlikte çalışan insanlar birbirlerine yakın oluyorlar.
Yes : kasmayin artik ya kac kere tanik olduk bu azgin tehlikeli “multecilerin” yaptiklarina? bir afgan taragindan kafasi tasla ezilip tecavuz edilen kiza da git boyle cihangir solculugu yap yerse?
`,q="python run_hatespeech.py",D=m({__name:"HateSpeechMain",setup(c){const o=String.raw`#Linux/Mac users
source venv/bin/activate
# Windows users
.\venv\Scripts\activate
`;return(s,l)=>{const i=p,a=g,_=h;return r(),u("main",y,[e(_,null,{default:n(()=>[t("div",null,[T,v,w,f(" To begin using the TurkishBERTweet model for sentiment analysis, you'll first need to set up your development environment. This involves cloning the TurkishBERTweet repository, creating a virtual environment, and installing the necessary libraries. Follow these steps to get started: "),x,B,E,e(a,null,{default:n(()=>[e(i,{code:V,language:"bash",filename:"clone.sh"})]),_:1}),R,e(a,null,{default:n(()=>[e(i,{code:N,language:"bash",filename:"cd.sh"})]),_:1}),z,e(a,null,{default:n(()=>[e(i,{code:P,language:"bash",filename:"venv.sh"})]),_:1}),S,e(a,null,{default:n(()=>[e(i,{code:o,language:"bash",filename:"activate.sh"})]),_:1}),C,e(a,null,{default:n(()=>[e(i,{code:A,language:"text",filename:"install.sh"})]),_:1}),H,e(a,null,{default:n(()=>[e(i,{code:q,language:"bash",filename:"run_finetune.sh"})]),_:1}),e(a,null,{default:n(()=>[e(i,{code:L,language:"python",filename:"run_hatespeech.py"})]),_:1}),e(a,null,{default:n(()=>[e(i,{code:F,filename:"output.txt"})]),_:1})]),M]),_:1})])}}}),I={};function U(c,o){const s=d,l=D;return r(),u(b,null,[e(s,{title:"Hate Speech Detection"}),e(l)],64)}const j=k(I,[["render",U]]);export{j as default};
