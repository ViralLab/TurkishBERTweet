import{_ as u,o as s,c,p as g,e as b,a as e,f as h,r as x,g as $,h as v,i as m,j as C,b as o,k as y,l as L,w as p,m as k}from"./BYfiZp86.js";import{_ as w}from"./BNGvvF6H.js";import"./UFG5L2p4.js";const H={},d=t=>(g("data-v-772b5e97"),t=t(),b(),t),I={class:"header-background flex flex-col items-center justify-center space-y-4 bg-cover bg-center bg-no-repeat py-14 text-gray-200"},E=d(()=>e("h1",{class:"text-4xl"},"Reasercher",-1)),B=d(()=>e("hr",{class:"w-1/12"},null,-1)),M=d(()=>e("p",{class:"text-xl"}," Lorem ipsum dolor sit amet consectetur adipisicing elit. ",-1)),S=[E,B,M];function V(t,n){return s(),c("header",I,S)}const j=u(H,[["render",V],["__scopeId","data-v-772b5e97"]]),A={class:"relative"},N={class:"absolute right-4 top-4 flex items-center gap-2"},T={key:0,class:"text-primary text-sm"},U=h({__name:"CodeHighlight",props:{code:{},language:{}},setup(t){const n=t,a=x(!1);function l(){n.code&&navigator.clipboard.writeText(n.code),a.value=!0}return(r,_)=>{const i=$("HightlightJs"),f=w;return s(),c("div",A,[r.code?(s(),v(i,{key:0,code:r.code,autodetect:!r.language,language:r.language},null,8,["code","autodetect","language"])):m("",!0),e("div",N,[C(a)?(s(),c("span",T,"Copied!")):m("",!0),e("button",{onClick:l,onMouseleave:_[0]||(_[0]=le=>a.value=!1)},[o(f,{name:"i-heroicons-clipboard-document-check-solid",class:"hover:bg-primary h-6 w-6 bg-background"})],32)])])}}}),q={},J={class:"custom-html-content"};function P(t,n){return s(),c("div",J,[y(t.$slots,"default")])}const Q=u(q,[["render",P]]),R=L("/images/laptop.jpg"),O={},z={class:"border-border w-80 border p-2"},D=e("a",{href:"/"},[e("img",{src:R,alt:"laptop image"}),e("div",{class:"mt-2 text-start"},[e("h5",null,"Lorem, ipsum."),e("p",null,"Lorem ipsum dolor sit amet.")])],-1),F=[D];function G(t,n){return s(),c("div",z,F)}const K=u(O,[["render",G]]),W={class:"inner pt-10"},X=e("h2",null,"Lorem ipsum dolor sit ?",-1),Y=e("p",null," Lorem ipsum dolor sit amet consectetur adipisicing elit. Voluptate repudiandae reprehenderit accusantium similique ipsa? ",-1),Z=e("p",null," Lorem ipsum dolor sit amet consectetur adipisicing elit. Quibusdam laudantium necessitatibus beatae. In, officiis nostrum autem ea minima ipsum numquam. ",-1),ee=e("p",null," Lorem ipsum dolor, sit amet consectetur adipisicing elit. Eum at pariatur dolor? Quibusdam. ",-1),te=e("br",null,null,-1),oe=e("div",null,[e("h2",null,"Lorem, ipsum dolor ?"),e("ul",null,[e("li",null," Lorem ipsum dolor sit amet consectetur adipisicing elit. Autem, sunt! "),e("li",null," Lorem ipsum dolor sit amet consectetur adipisicing elit. Molestias et amet cum tempora ad. "),e("li",null,"Lorem ipsum dolor sit amet consectetur, adipisicing elit. Ad!")])],-1),se=e("br",null,null,-1),ne={class:"flex flex-wrap gap-8"},ie=`# Python Program to find the area of triangle

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
print('The area of the triangle is %0.2f' %area)`,ce=h({__name:"HomeMain",setup(t){return(n,a)=>{const l=U,r=k,_=Q,i=K;return s(),c("main",W,[o(_,null,{default:p(()=>[e("div",null,[X,o(r,null,{default:p(()=>[o(l,{code:ie,language:"python"})]),_:1}),Y,Z,ee]),te,oe]),_:1}),se,e("div",ne,[o(i),o(i),o(i),o(i),o(i)])])}}}),ae={};function re(t,n){const a=j,l=ce;return s(),c("div",null,[o(a),o(l)])}const me=u(ae,[["render",re]]);export{me as default};
