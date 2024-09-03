import{_ as d,o as r,c as l,p as h,e as m,a as t,f,b as e,w as n,d as g,g as b,F as k}from"./Cl6uyuXL.js";import{_ as y}from"./D5hmElPJ.js";import{_ as v}from"./VaQbr8R-.js";import"./tNQpnWyy.js";import"./sYAdBX6R.js";const T={},_=s=>(h("data-v-2b662a4b"),s=s(),m(),s),w={class:"header-background flex flex-col items-center justify-center space-y-8 bg-cover bg-center bg-no-repeat py-10 md:py-16 lg:py-24 text-gray-200"},x=_(()=>t("h1",{class:"text-6xl"},"Sentiment Analysis",-1)),E=_(()=>t("hr",{class:"w-1/12"},null,-1)),B=_(()=>t("p",{class:"text-2xl"},null,-1)),R=[x,E,B];function S(s,a){return r(),l("header",w,R)}const z=d(T,[["render",S],["__scopeId","data-v-2b662a4b"]]),A={class:"inner pt-10"},C=t("h3",null,"Sentiment Analysis with TurkishBERTweet",-1),M=t("p",null,"In this section, we will guide you through the process of using TurkishBERTweet for sentiment analysis. TurkishBERTweet is a specialized model designed to analyze Turkish social media text, enabling you to determine the sentiment behind tweets, posts, and comments with high accuracy. ",-1),$=t("h3",null,"Setting Up the Environment for TurkishBERTweet",-1),V=t("br",null,null,-1),I=t("br",null,null,-1),L=t("h5",null,"1. Clone the Repository: Begin by cloning the TurkishBERTweet repository from GitHub.",-1),P=t("h5",null,"2. Navigate to the Directory: Move into the newly cloned directory.",-1),j=t("h5",null,"3. Set Up a Virtual Environment: Create a virtual environment to manage dependencies.",-1),F=t("h5",null,"4. Activate the Virtual Environment: Activate the virtual environment.",-1),H=t("h5",null,"5. Install Required Libraries: Install PyTorch and other essential libraries to run TurkishBERTweet.",-1),N=t("h5",null,"6. Run the following script to get the sentiments of the sample texts:",-1),q=t("br",null,null,-1),U="git clone git@github.com:ViralLab/TurkishBERTweet.git",D="cd TurkishBERTweet",G="python -m venv venv",O=`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install peft
pip install transformers
pip install urlextract`,W=`import torch
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
`,Z=`positive : viral lab da insanlar hep birlikte çalışıyorlar. hepbirlikte çalışan insanlar birbirlerine yakın oluyorlar.
negative : americanin diplatlari turkiyeye gelmesin <emoji> burundan_buharla_yüzleşmek </emoji>
positive : mark zuckerberg ve elon musk'un boks müsabakası süper olacak! <emoji> kadın_muhafız_koyu_ten_tonu </emoji>
neutral : adam dun ne yediğini unuttu`,J="python run_sentiment.py",K=f({__name:"SentimentMain",setup(s){const a=String.raw`#Linux/Mac users
source venv/bin/activate
# Windows users
.\venv\Scripts\activate
`;return(u,c)=>{const o=y,i=b,p=v;return r(),l("main",A,[e(p,null,{default:n(()=>[t("div",null,[C,M,$,g(" To begin using the TurkishBERTweet model for sentiment analysis, you'll first need to set up your development environment. This involves cloning the TurkishBERTweet repository, creating a virtual environment, and installing the necessary libraries. Follow these steps to get started: "),V,I,L,e(i,null,{default:n(()=>[e(o,{code:U,language:"bash",filename:"clone.sh"})]),_:1}),P,e(i,null,{default:n(()=>[e(o,{code:D,language:"bash",filename:"cd.sh"})]),_:1}),j,e(i,null,{default:n(()=>[e(o,{code:G,language:"bash",filename:"venv.sh"})]),_:1}),F,e(i,null,{default:n(()=>[e(o,{code:a,language:"bash",filename:"activate.sh"})]),_:1}),H,e(i,null,{default:n(()=>[e(o,{code:O,language:"text",filename:"install.sh"})]),_:1}),N,e(i,null,{default:n(()=>[e(o,{code:J,language:"bash",filename:"run_finetune.sh"})]),_:1}),e(i,null,{default:n(()=>[e(o,{code:W,language:"python",filename:"run_sentiment.py"})]),_:1}),e(i,null,{default:n(()=>[e(o,{code:Z,filename:"output.txt"})]),_:1})]),q]),_:1})])}}}),Q={};function X(s,a){const u=z,c=K;return r(),l(k,null,[e(u),e(c)],64)}const ie=d(Q,[["render",X]]);export{ie as default};
