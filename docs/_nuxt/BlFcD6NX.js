import{_,o as n,c as a,p as g,e as b,a as e,f as h,r as x,g as $,h as v,i as p,b as t,w as d,j as C,T as y,k as L,l as k,m as w}from"./BykPfcpU.js";import{_ as H}from"./Bdrl6Fba.js";import"./DHYNZlBB.js";const I={},m=o=>(g("data-v-772b5e97"),o=o(),b(),o),E={class:"header-background flex flex-col items-center justify-center space-y-4 bg-cover bg-center bg-no-repeat py-14 text-gray-200"},T=m(()=>e("h1",{class:"text-4xl"},"Reasercher",-1)),B=m(()=>e("hr",{class:"w-1/12"},null,-1)),M=m(()=>e("p",{class:"text-xl"}," Lorem ipsum dolor sit amet consectetur adipisicing elit. ",-1)),S=[T,B,M];function V(o,s){return n(),a("header",E,S)}const j=_(I,[["render",V],["__scopeId","data-v-772b5e97"]]),A={class:"relative"},N={class:"absolute right-4 top-4 flex items-center gap-2"},U={key:0,class:"text-primary text-sm"},q=h({__name:"CodeHighlight",props:{code:{},language:{}},setup(o){const s=o,c=x(!1);function l(){s.code&&navigator.clipboard.writeText(s.code),c.value=!0}return(r,u)=>{const i=$("HightlightJs"),f=H;return n(),a("div",A,[r.code?(n(),v(i,{key:0,code:r.code,autodetect:!r.language,language:r.language},null,8,["code","autodetect","language"])):p("",!0),e("div",N,[t(y,null,{default:d(()=>[C(c)?(n(),a("span",U,"Copied!")):p("",!0)]),_:1}),e("button",{onClick:l,onMouseleave:u[0]||(u[0]=ue=>c.value=!1)},[t(f,{name:"i-heroicons-clipboard-document-check-solid",class:"hover:bg-primary h-6 w-6 bg-background"})],32)])])}}}),J=_(q,[["__scopeId","data-v-3707312e"]]),P={},Q={class:"custom-html-content"};function R(o,s){return n(),a("div",Q,[L(o.$slots,"default")])}const O=_(P,[["render",R]]),z=k("/images/laptop.jpg"),D={},F={class:"border-border w-80 border p-2"},G=e("a",{href:"/"},[e("img",{src:z,alt:"laptop image"}),e("div",{class:"mt-2 text-start"},[e("h5",null,"Lorem, ipsum."),e("p",null,"Lorem ipsum dolor sit amet.")])],-1),K=[G];function W(o,s){return n(),a("div",F,K)}const X=_(D,[["render",W]]),Y={class:"inner pt-10"},Z=e("h2",null,"Lorem ipsum dolor sit ?",-1),ee=e("p",null," Lorem ipsum dolor sit amet consectetur adipisicing elit. Voluptate repudiandae reprehenderit accusantium similique ipsa? ",-1),te=e("p",null," Lorem ipsum dolor sit amet consectetur adipisicing elit. Quibusdam laudantium necessitatibus beatae. In, officiis nostrum autem ea minima ipsum numquam. ",-1),oe=e("p",null," Lorem ipsum dolor, sit amet consectetur adipisicing elit. Eum at pariatur dolor? Quibusdam. ",-1),ne=e("br",null,null,-1),se=e("div",null,[e("h2",null,"Lorem, ipsum dolor ?"),e("ul",null,[e("li",null," Lorem ipsum dolor sit amet consectetur adipisicing elit. Autem, sunt! "),e("li",null," Lorem ipsum dolor sit amet consectetur adipisicing elit. Molestias et amet cum tempora ad. "),e("li",null,"Lorem ipsum dolor sit amet consectetur, adipisicing elit. Ad!")])],-1),ie=e("br",null,null,-1),ae={class:"flex flex-wrap gap-8"},ce=`# Python Program to find the area of triangle

a = 5
b = 6
c = 7

# Uncomment below to take inputs from the user
# a = float(input('Enter first side: '))
# b = float(input('Enter second side: '))
# c = float(input('Enter third side: '))

# calculate the semi-perimeter
s = (a + b + c) / 2

# calculate the area
area = (s*(s-a)*(s-b)*(s-c)) ** 0.5
print('The area of the triangle is %0.2f' %area)`,re=h({__name:"HomeMain",setup(o){return(s,c)=>{const l=J,r=w,u=O,i=X;return n(),a("main",Y,[t(u,null,{default:d(()=>[e("div",null,[Z,t(r,null,{default:d(()=>[t(l,{code:ce,language:"python"})]),_:1}),ee,te,oe]),ne,se]),_:1}),ie,e("div",ae,[t(i),t(i),t(i),t(i),t(i)])])}}}),le={};function _e(o,s){const c=j,l=re;return n(),a("div",null,[t(c),t(l)])}const he=_(le,[["render",_e]]);export{he as default};
