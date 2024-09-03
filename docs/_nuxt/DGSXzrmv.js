import{_,o as s,c as a,p as m,e as g,a as e,t as p,n as y,f as x,b as n,w as u,d as h,g as $,F as v}from"./Df_DkY5G.js";import{_ as S}from"./D_qlBDOx.js";import{_ as k}from"./BgVM38eb.js";import"./BgZ1xh1b.js";import"./CU7MWXEm.js";const C={},d=t=>(m("data-v-a24c0661"),t=t(),g(),t),j={class:"header-background flex flex-col items-center justify-center space-y-8 bg-cover bg-center bg-no-repeat py-10 md:py-16 lg:py-24 text-gray-200"},A=d(()=>e("h1",{class:"text-6xl"},"ABOUT US",-1)),T=d(()=>e("hr",{class:"w-1/12"},null,-1)),I=d(()=>e("p",{class:"text-2xl"},null,-1)),q=[A,T,I];function H(t,r){return s(),a("header",j,q)}const B=_(C,[["render",H],["__scopeId","data-v-a24c0661"]]),N={class:"border-border w-60 border p-2 rounded-md"},U=["href"],V=["src"],E={class:"mt-2 text-center"},F={__name:"HomeCard",props:{name:{type:String,required:!0},title:{type:String,required:!0},pic:{type:String,required:!0},webpage:{type:String,required:!0}},setup(t){return(r,i)=>(s(),a("div",N,[e("a",{href:t.webpage,target:"_blank"},[e("img",{src:`images/${t.pic}`,alt:"laptop image",class:"rounded-md"},null,8,V)],8,U),e("div",E,[e("h4",null,p(t.name),1),e("p",null,p(t.title),1)])]))}},O=["href"],M=["src"],P={__name:"Logo",props:{pic:{type:String,required:!0},webpage:{type:String,required:!0},mywidth:{type:String,required:!0}},setup(t){return(r,i)=>(s(),a("div",{class:"border-border border p-2 rounded-md",style:y({width:`${t.mywidth}px`})},[e("a",{href:t.webpage,target:"_blank"},[e("img",{src:`images/${t.pic}`,alt:"logo",class:"rounded-md"},null,8,M)],8,O)],4))}},o=t=>(m("data-v-f64fe67d"),t=t(),g(),t),W={class:"inner pt-10"},G=o(()=>e("h1",null,"Who are we?",-1)),K={class:"flex flex-wrap gap-8 items-center"},L=o(()=>e("br",null,null,-1)),R=o(()=>e("br",null,null,-1)),z=o(()=>e("br",null,null,-1)),D=o(()=>e("h2",null,"How to Cite",-1)),J=o(()=>e("div",{class:"specialfont"},"TurkishBERTweet",-1)),Q=o(()=>e("br",null,null,-1)),X=o(()=>e("br",null,null,-1)),Y=o(()=>e("h2",null,"Supporting Projects",-1)),Z=o(()=>e("p",null," We thank Fatih Amasyali for providing access to Tweet Sentiment datasets from Kemik group. This material is based upon work supported by the Google Cloud Research Credits program with the award GCP19980904. We also thank TUBITAK (121C220 and 222N311) for funding this project. ",-1)),ee=o(()=>e("br",null,null,-1)),te={class:"flex flex-wrap gap-8 items-center"},oe=o(()=>e("br",null,null,-1)),ne=`@article{najafi2024turkishbertweet,
  title={Turkishbertweet: Fast and reliable large language model for social media analysis},
  author={Najafi, Ali and Varol, Onur},
  journal={Expert Systems with Applications},
  volume={255},
  pages={124737},
  year={2024},
  publisher={Elsevier}
}
`,se=x({__name:"AboutMain",setup(t){return(r,i)=>{const c=F,f=S,b=$,l=P,w=k;return s(),a("main",W,[n(w,null,{default:u(()=>[e("div",null,[G,e("div",K,[n(c,{name:"Ali Najafi",title:"MS.C. Computer Science @Sabanci University",pic:"ali_najafi.jpg",webpage:"http://najafi-ali.com"}),n(c,{name:"Onur Varol",title:"Assistant Professor @Sabanci University",pic:"onur_varol.jpg",webpage:"http://www.onurvarol.com/"})]),L,R,z,D,h(" If you use "),J,h(" in your research, please cite the following paper: "),n(b,null,{default:u(()=>[n(f,{code:ne,filename:"cite.txt"})]),_:1}),Q,X,Y,Z,ee,e("div",te,[n(l,{pic:"tubitak-logo.png",webpage:"https://tubitak.gov.tr/en",mywidth:"120"}),n(l,{pic:"sabanci-logo.png",webpage:"https://sabanciuniv.edu/",mywidth:"250"}),n(l,{pic:"google_research.jpg",webpage:"https://sites.research.google/trc/about/",mywidth:"300"})])])]),_:1}),oe])}}}),ae=_(se,[["__scopeId","data-v-f64fe67d"]]),re={};function ie(t,r){const i=B,c=ae;return s(),a(v,null,[n(i),n(c)],64)}const ue=_(re,[["render",ie]]);export{ue as default};
