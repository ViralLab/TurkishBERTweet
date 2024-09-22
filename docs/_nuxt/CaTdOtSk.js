import{_ as p,a as m}from"./DXHoEFTA.js";import{a as g,_ as T}from"./BO_qo4DQ.js";import{_ as c,o as l,c as r,a as e,t as f,p as h,e as d,b as s,w as k}from"./DAmhm2ud.js";import"./CQSScytE.js";import"./Ovsnsdwn.js";import"./C1v8UIkE.js";import"./Dg_zS6C8.js";const w=t=>(h("data-v-d6e1d9d6"),t=t(),d(),t),x={class:"flex items-center space-x-3 rtl:space-x-reverse"},b=w(()=>e("svg",{xmlns:"http://www.w3.org/2000/svg",width:"18",height:"18",viewBox:"0 0 24 24",class:"text-primary"},[e("path",{fill:"currentColor",d:"M12 22q-2.075 0-3.9-.788t-3.175-2.137T2.788 15.9T2 12t.788-3.9t2.137-3.175T8.1 2.788T12 2t3.9.788t3.175 2.137T21.213 8.1T22 12t-.788 3.9t-2.137 3.175t-3.175 2.138T12 22m0-2q3.35 0 5.675-2.325T20 12q0-.175-.012-.363t-.013-.312q-.125.725-.675 1.2T18 13h-2q-.825 0-1.412-.587T14 11v-1h-4V8q0-.825.588-1.412T12 6h1q0-.575.313-1.012t.762-.713q-.5-.125-1.012-.2T12 4Q8.65 4 6.325 6.325T4 12h5q1.65 0 2.825 1.175T13 16v1h-3v2.75q.5.125.988.188T12 20"})],-1)),R=["href"],B={__name:"MyListitem",props:{link:{type:String,required:!0},text:{type:String,required:!0}},setup(t){return(_,o)=>(l(),r("div",null,[e("li",x,[b,e("a",{class:"menu",href:t.link,target:"_blank"},f(t.text),9,R)])]))}},v=c(B,[["__scopeId","data-v-d6e1d9d6"]]),L={},n=t=>(h("data-v-f000ed25"),t=t(),d(),t),y={class:"inner pt-10"},E=n(()=>e("br",null,null,-1)),S=n(()=>e("h2",null,"What is TurkishBERTweet?",-1)),q=n(()=>e("p",null," TurkishBERTweet is the first large-scale pre-trained language model specifically designed for Turkish social media. Built using over 894 million Turkish tweets, it shares the architecture of the RoBERTa-base model but is optimized for social media content. TurkishBERTweet excels in tasks like Sentiment Classification and Hate Speech Detection, offering better generalizability and lower inference times compared to existing models. ",-1)),$=n(()=>e("h2",null,"Why TurkishBERTweet?",-1)),H=n(()=>e("p",null," Despite Turkish being widely spoken, it is still considered a low-resource language in NLP. With Turkey’s strategic role in global politics and the vast amount of Turkish content on platforms like Twitter and Instagram, there is a growing need for tools tailored to this language. TurkishBERTweet addresses this by offering a cost-effective, scalable solution for processing Turkish social media data, outperforming other models and commercial solutions like OpenAI. ",-1)),I=n(()=>e("h2",{class:"pb-4"},"Available Models on HuggingFace 🤗",-1)),M={class:"space-y-4 text-left"},A=n(()=>e("br",null,null,-1)),V=n(()=>e("br",null,null,-1)),C=n(()=>e("br",null,null,-1)),D=n(()=>e("h2",null,"Useful Links",-1)),j={class:"useful-links grid gap-4"};function F(t,_){const o=g,i=v,a=T,u=p;return l(),r("main",y,[s(u,null,{default:k(()=>[e("div",null,[s(o),E,S,q,$,H,I,e("ul",M,[s(i,{link:"https://huggingface.co/VRLLab/TurkishBERTweet",text:"TurkishBERTweet Language Model"}),s(i,{link:"https://huggingface.co/VRLLab/TurkishBERTweet-Lora-SA",text:"TurkishBERTweet Sentiment LoRA"}),s(i,{link:"https://huggingface.co/VRLLab/TurkishBERTweet-Lora-HS",text:"TurkishBERTweet Hate Speech LoRA"})]),A,V,C,D,e("div",j,[s(a,{name:"Finetuning TurkishBERTweet",title:"",pic:"lora.gif",webpage:"/finetune"}),s(a,{name:"Sentiment Analysis",title:"",pic:"sentiment.jpg",webpage:"/sentimentanalysis"}),s(a,{name:"HateSpeech Detection",title:"",pic:"hatespeech.jpeg",webpage:"/hatespeech"})])])]),_:1})])}const N=c(L,[["render",F],["__scopeId","data-v-f000ed25"]]),W={};function z(t,_){const o=m,i=N;return l(),r("div",null,[s(o,{pic:"/TurkishBERTweet.jpg",title:"TurkishBERTweet",description:"Fast and Reliable Large Language Model for Social Media Analysis"}),s(i)])}const X=c(W,[["render",z]]);export{X as default};