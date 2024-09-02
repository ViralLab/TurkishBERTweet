import{_ as m,o as s,c as i,p as u,e as d,a as e,h,b as t,w as n,d as x,i as f,F as g}from"./CYpyroB7.js";import{_ as b}from"./DeUhWyqh.js";import{_ as k}from"./d-XrPoj7.js";import"./D6-TuxIk.js";import"./YkhGPo3c.js";const v={},c=o=>(u("data-v-96987dbc"),o=o(),d(),o),y={class:"header-background flex flex-col items-center justify-center space-y-8 bg-cover bg-center bg-no-repeat py-10 md:py-16 lg:py-24 text-gray-200"},H=c(()=>e("h1",{class:"text-6xl"},"Hate Speech Detection",-1)),L=c(()=>e("hr",{class:"w-1/12"},null,-1)),$=c(()=>e("p",{class:"text-2xl"},null,-1)),S=[H,L,$];function C(o,r){return s(),i("header",y,S)}const I=m(v,[["render",C],["__scopeId","data-v-96987dbc"]]),V={class:"inner pt-10"},w=e("h2",null,"Preprocessor",-1),z=e("p",null," Lorem ipsum dolor sit amet consectetur adipisicing elit. Voluptate repudiandae reprehenderit accusantium similique ipsa? ",-1),j=e("p",null," Lorem ipsum dolor sit amet consectetur adipisicing elit. Quibusdam laudantium necessitatibus beatae. In, officiis nostrum autem ea minima ipsum numquam. ",-1),B=e("p",null," Lorem ipsum dolor, sit amet consectetur adipisicing elit. Eum at pariatur dolor? Quibusdam. ",-1),M=e("br",null,null,-1),N=e("div",null,[e("h2",null,"Lorem, ipsum dolor ?"),e("ul",null,[e("li",null," Lorem ipsum dolor sit amet consectetur adipisicing elit. Autem, sunt! "),e("li",null," Lorem ipsum dolor sit amet consectetur adipisicing elit. Molestias et amet cum tempora ad. "),e("li",null,"Lorem ipsum dolor sit amet consectetur, adipisicing elit. Ad!")])],-1),q=`
from Preprocessor import preprocess

text = """Lab'ımıza "viral" adını verdik çünkü amacımız disiplinler arası sınırları aşmak ve aralarında yeni bağlantılar kurmak! 🔬 #ViralLab
https://varollab.com/"""

preprocessed_text = preprocess(text)
print(preprocessed_text)
`,A=`
lab'ımıza "viral" adını verdik çünkü amacımız disiplinler arası sınırları aşmak ve aralarında yeni bağlantılar kurmak! <emoji> mikroskop </emoji> <hashtag> virallab </hashtag> <http> varollab.com </http>
`,E=h({__name:"HateSpeechMain",setup(o){return(r,l)=>{const a=b,p=f,_=k;return s(),i("main",V,[t(_,null,{default:n(()=>[e("div",null,[w,t(p,null,{default:n(()=>[t(a,{code:q,language:"python"})]),_:1}),x(" Output: "),t(p,null,{default:n(()=>[t(a,{code:A,language:"text"})]),_:1}),z,j,B]),M,N]),_:1})])}}}),F={};function O(o,r){const l=I,a=E;return s(),i(g,null,[t(l),t(a)],64)}const J=m(F,[["render",O]]);export{J as default};
