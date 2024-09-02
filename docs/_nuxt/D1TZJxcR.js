import{_ as d,o as n,c as s,p as u,e as m,a as e,f as h,b as t,w as a,d as f,g,F as k}from"./CSoyElds.js";import{_ as b}from"./eAXRnBeW.js";import{_ as x}from"./9MOn9I1i.js";import"./Dtdg3njw.js";import"./DRKT6Zaj.js";const y={},r=o=>(u("data-v-96987dbc"),o=o(),m(),o),z={class:"header-background flex flex-col items-center justify-center space-y-8 bg-cover bg-center bg-no-repeat py-10 md:py-16 lg:py-24 text-gray-200"},v=r(()=>e("h1",{class:"text-6xl"},"Hate Speech Detection",-1)),S=r(()=>e("hr",{class:"w-1/12"},null,-1)),T=r(()=>e("p",{class:"text-2xl"},null,-1)),w=[v,S,T];function H(o,l){return n(),s("header",z,w)}const L=d(y,[["render",H],["__scopeId","data-v-96987dbc"]]),C={class:"inner pt-10"},$=e("p",null," Lorem ipsum dolor sit amet consectetur adipisicing elit. Voluptate repudiandae reprehenderit accusantium similique ipsa? ",-1),B=e("p",null," Lorem ipsum dolor sit amet consectetur adipisicing elit. Quibusdam laudantium necessitatibus beatae. In, officiis nostrum autem ea minima ipsum numquam. ",-1),E=e("p",null," Lorem ipsum dolor, sit amet consectetur adipisicing elit. Eum at pariatur dolor? Quibusdam. ",-1),M=e("br",null,null,-1),A=e("div",null,[e("h2",null,"Lorem, ipsum dolor ?"),e("ul",null,[e("li",null," Lorem ipsum dolor sit amet consectetur adipisicing elit. Autem, sunt! "),e("li",null," Lorem ipsum dolor sit amet consectetur adipisicing elit. Molestias et amet cum tempora ad. "),e("li",null,"Lorem ipsum dolor sit amet consectetur, adipisicing elit. Ad!")])],-1),R=`from peft import (
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
`,V=`
lab'ımıza "viral" adını verdik çünkü amacımız disiplinler arası sınırları aşmak ve aralarında yeni bağlantılar kurmak! <emoji> mikroskop </emoji> <hashtag> virallab </hashtag> <http> varollab.com </http>
`,I=h({__name:"HateSpeechMain",setup(o){return(l,c)=>{const i=b,_=g,p=x;return n(),s("main",C,[t(p,null,{default:a(()=>[e("div",null,[t(_,null,{default:a(()=>[t(i,{code:R,language:"python"})]),_:1}),f(" Output: "),t(_,null,{default:a(()=>[t(i,{code:V,language:"text"})]),_:1}),$,B,E]),M,A]),_:1})])}}}),N={};function P(o,l){const c=L,i=I;return n(),s(k,null,[t(c),t(i)],64)}const D=d(N,[["render",P]]);export{D as default};
