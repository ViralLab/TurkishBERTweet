import{_}from"./BgZ1xh1b.js";import{f as h,r as g,i as f,o as l,c as d,j as b,k as p,a as t,t as v,b as i,w as k,l as y,T as w,p as C,e as x,_ as B}from"./Df_DkY5G.js";const I=a=>(C("data-v-bcf9c741"),a=a(),x(),a),U={class:"relative"},L={class:"absolute left-4 top-2 flex items-center gap-2"},R={class:"fname_style"},S=I(()=>t("br",null,null,-1)),T={class:"absolute right-4 top-4 flex items-center gap-2"},j={key:0,class:"text-primary text-sm"},H=h({__name:"CodeHighlight",props:{code:{},language:{},filename:{}},setup(a){const n=a,c=g(!1);function u(){n.code&&navigator.clipboard.writeText(n.code),c.value=!0}function m(){if(n.code){const e=new Blob([n.code],{type:"text/x-python"}),s=URL.createObjectURL(e),o=document.createElement("a");o.href=s,o.download=n.filename,document.body.appendChild(o),o.click(),document.body.removeChild(o),URL.revokeObjectURL(s)}}return(e,s)=>{const o=f("HightlightJs"),r=_;return l(),d("div",U,[e.code?(l(),b(o,{key:0,code:`
${e.code}`,autodetect:!e.language,language:e.language},null,8,["code","autodetect","language"])):p("",!0),t("div",L,[t("span",R,v(e.filename),1),S]),t("div",T,[i(w,null,{default:k(()=>[y(c)?(l(),d("span",j,"Copied!")):p("",!0)]),_:1}),t("button",{onClick:u,onMouseleave:s[0]||(s[0]=N=>c.value=!1)},[i(r,{name:"i-heroicons-clipboard-document-check-solid",class:"hover:bg-primary h-6 w-6 bg-background"})],32),t("button",{onClick:m},[i(r,{name:"i-heroicons-arrow-down-tray",class:"hover:bg-primary h-6 w-6 bg-background"})])])])}}}),E=B(H,[["__scopeId","data-v-bcf9c741"]]);export{E as _};
