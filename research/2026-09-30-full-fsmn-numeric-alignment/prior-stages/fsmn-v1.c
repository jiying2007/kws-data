#include "fsmn.h"
#include <math.h>
#include <string.h>
static void affine(const float*x,float*y,const float*w,const float*b,int ni,int no){
 for(int o=0;o<no;o++){float s=0;for(int i=0;i<ni;i++)s+=x[i]*w[o*ni+i];y[o]=s+(b?b[o]:0);}
}
static void relu(float*x,int n){for(int i=0;i<n;i++)if(x[i]<0)x[i]=0;}
static void trace(float*t,int s,const float*x,int n){if(t)memcpy(t+(size_t)s*2599,x,(size_t)n*sizeof(float));}
int df_init(df_model*m,const float*w,size_t n){
 if(!m||!w||n!=DF_FLOATS)return -1;
 for(size_t i=0;i<n;i++)if(!isfinite(w[i]))return -2;
 m->weights=w;df_reset(m);return 0;
}
void df_reset(df_model*m){if(m)memset(m->cache,0,sizeof(m->cache));}
int df_step(df_model*m,const float*x,float*y,float*t){
 if(!m||!m->weights||!x||!y)return -1;
 for(int i=0;i<400;i++)if(!isfinite(x[i]))return -2;
 const float*w=m->weights;float z[400],a[250],b[250],p[128],mem[128];
 for(int i=0;i<400;i++)z[i]=(x[i]-w[i])*w[400+i];
 affine(z,a,w+800,w+56800,400,140);trace(t,0,a,140);
 affine(a,b,w+56940,w+91940,140,250);trace(t,1,b,250);relu(b,250);trace(t,2,b,250);
 for(int l=0;l<4;l++){
  size_t off=92190u+(size_t)l*65786u;
  affine(b,p,w+off,0,250,128);trace(t,3+4*l,p,128);
  for(int c=0;c<128;c++){
   float left=0;for(int k=0;k<10;k++)left+=w[off+32000u+(size_t)c*10+k]*m->cache[((size_t)c*11+k)*4+l];
   float right=w[off+33280u+(size_t)c*2]*m->cache[((size_t)c*11+10)*4+l]+w[off+33281u+(size_t)c*2]*p[c];
   mem[c]=(m->cache[((size_t)c*11+9)*4+l]+left)+right;
   for(int k=0;k<10;k++)m->cache[((size_t)c*11+k)*4+l]=m->cache[((size_t)c*11+k+1)*4+l];
   m->cache[((size_t)c*11+10)*4+l]=p[c];
  }
  trace(t,4+4*l,mem,128);affine(mem,b,w+off+33536,w+off+65536,128,250);trace(t,5+4*l,b,250);relu(b,250);trace(t,6+4*l,b,250);
 }
 affine(b,a,w+355334,w+390334,250,140);trace(t,19,a,140);
 affine(a,y,w+390474,w+754334,140,2599);trace(t,20,y,2599);
 for(int i=0;i<2599;i++)if(!isfinite(y[i]))return -3;
 return 0;
}
