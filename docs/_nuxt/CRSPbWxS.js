import{_ as d,a as p}from"./DXHoEFTA.js";import{_ as m}from"./BtIylxeA.js";import{g as h,o as l,c as u,b as e,w as n,a as t,d as f,h as g,_ as k,F as b}from"./DAmhm2ud.js";import"./C1v8UIkE.js";import"./Dg_zS6C8.js";const y={class:"inner pt-10"},v=t("h3",null,"Sentiment Analysis with TurkishBERTweet",-1),T=t("p",null,"In this section, we will guide you through the process of using TurkishBERTweet for sentiment analysis. TurkishBERTweet is a specialized model designed to analyze Turkish social media text, enabling you to determine the sentiment behind tweets, posts, and comments with high accuracy. ",-1),w=t("h3",null,"Setting Up the Environment for TurkishBERTweet",-1),x=t("br",null,null,-1),E=t("br",null,null,-1),B=t("h5",null,"1. Clone the Repository: Begin by cloning the TurkishBERTweet repository from GitHub.",-1),R=t("h5",null,"2. Navigate to the Directory: Move into the newly cloned directory.",-1),z=t("h5",null,"3. Set Up a Virtual Environment: Create a virtual environment to manage dependencies.",-1),S=t("h5",null,"4. Activate the Virtual Environment: Activate the virtual environment.",-1),A=t("h5",null,"5. Install Required Libraries: Install PyTorch and other essential libraries to run TurkishBERTweet.",-1),C=t("h5",null,"6. Run the following script to get the sentiments of the sample texts:",-1),M=t("br",null,null,-1),V="git clone git@github.com:ViralLab/TurkishBERTweet.git",L="cd TurkishBERTweet",P="python -m venv venv",F=`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install peft
pip install transformers
pip install urlextract`,N=`import torch
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
`,j=`positive : viral lab da insanlar hep birlikte çalışıyorlar. hepbirlikte çalışan insanlar birbirlerine yakın oluyorlar.
negative : americanin diplatlari turkiyeye gelmesin <emoji> burundan_buharla_yüzleşmek </emoji>
positive : mark zuckerberg ve elon musk'un boks müsabakası süper olacak! <emoji> kadın_muhafız_koyu_ten_tonu </emoji>
neutral : adam dun ne yediğini unuttu`,H="python run_sentiment.py",q=h({__name:"SentimentMain",setup(_){const s=String.raw`#Linux/Mac users
source venv/bin/activate
# Windows users
.\venv\Scripts\activate
`;return(a,r)=>{const i=m,o=g,c=d;return l(),u("main",y,[e(c,null,{default:n(()=>[t("div",null,[v,T,w,f(" To begin using the TurkishBERTweet model for sentiment analysis, you'll first need to set up your development environment. This involves cloning the TurkishBERTweet repository, creating a virtual environment, and installing the necessary libraries. Follow these steps to get started: "),x,E,B,e(o,null,{default:n(()=>[e(i,{code:V,language:"bash",filename:"clone.sh"})]),_:1}),R,e(o,null,{default:n(()=>[e(i,{code:L,language:"bash",filename:"cd.sh"})]),_:1}),z,e(o,null,{default:n(()=>[e(i,{code:P,language:"bash",filename:"venv.sh"})]),_:1}),S,e(o,null,{default:n(()=>[e(i,{code:s,language:"bash",filename:"activate.sh"})]),_:1}),A,e(o,null,{default:n(()=>[e(i,{code:F,language:"text",filename:"install.sh"})]),_:1}),C,e(o,null,{default:n(()=>[e(i,{code:H,language:"bash",filename:"run_finetune.sh"})]),_:1}),e(o,null,{default:n(()=>[e(i,{code:N,language:"python",filename:"run_sentiment.py"})]),_:1}),e(o,null,{default:n(()=>[e(i,{code:j,filename:"output.txt"})]),_:1})]),M]),_:1})])}}}),I={};function U(_,s){const a=p,r=q;return l(),u(b,null,[e(a,{title:"Sentiment Analysis"}),e(r)],64)}const Z=k(I,[["render",U]]);export{Z as default};
