#define _POSIX_C_SOURCE 200809L
#include "kws_pipeline/kws.h"
#include "tool_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <inttypes.h>

static double micros(clockid_t clock) {
  struct timespec t;
  if (clock_gettime(clock, &t) != 0) { perror("clock_gettime"); exit(2); }
  return (double)t.tv_sec * 1000000.0 + (double)t.tv_nsec / 1000.0;
}
static int cmp(const void *a,const void *b) { double x=*(const double *)a,y=*(const double *)b;return (x>y)-(x<y); }
static double percentile(const double *a,size_t n,double p) {
  size_t i=(size_t)(p*(double)n);if(i==0u)i=1u;if(i>n)i=n;return a[i-1u];
}
int main(int argc,char **argv) {
  uint8_t *mb=NULL,*kb=NULL;size_t mn=0u,kn=0u;kws_model_t model;kws_keyword_pack_t pack;
  double t,read_us,parse_us,init_us,cpu_us=0.0,wall_sum=0.0;void *arena;kws_engine_t *engine=NULL;
  FILE *wav;uint32_t wav_bytes=0u;long offset=0L;int16_t *pcm;double *times;
  const unsigned repeats=10u;size_t count=0u;uint64_t hits=0u;
  if(argc!=4){fprintf(stderr,"usage: profile model keywords wav\n");return 2;}
  t=micros(CLOCK_MONOTONIC);
  if(!kws_tool_read_file(argv[1],&mb,&mn)||!kws_tool_read_file(argv[2],&kb,&kn))return 2;
  read_us=micros(CLOCK_MONOTONIC)-t;
  t=micros(CLOCK_MONOTONIC);
  if(kws_model_open(mb,mn,&model)!=KWS_OK||kws_keyword_pack_open(kb,kn,&model,&pack)!=KWS_OK)return 2;
  parse_us=micros(CLOCK_MONOTONIC)-t;
  t=micros(CLOCK_MONOTONIC);arena=malloc(kws_engine_required_bytes(&model));
  if(!arena||kws_engine_init(arena,kws_engine_required_bytes(&model),&model,NULL,&engine)!=KWS_OK||kws_engine_set_keyword_pack(engine,&pack)!=KWS_OK)return 2;
  init_us=micros(CLOCK_MONOTONIC)-t;
  wav=fopen(argv[3],"rb");if(!wav||!kws_tool_open_wav(wav,&wav_bytes,&offset)||wav_bytes==0u)return 2;
  pcm=malloc(wav_bytes);if(!pcm||fseek(wav,offset,SEEK_SET)!=0||fread(pcm,1u,wav_bytes,wav)!=wav_bytes)return 2;
  fclose(wav);
  const size_t samples=(size_t)wav_bytes/2u;
  const size_t blocks=(samples+319u)/320u;
  times=malloc(blocks*(size_t)repeats*sizeof(*times));if(!times)return 2;
  for(unsigned rep=0u;rep<=repeats;++rep){
    kws_engine_reset(engine);
    double cpu_begin=micros(CLOCK_PROCESS_CPUTIME_ID);
    for(size_t pos=0u;pos<samples;pos+=320u){
      size_t n=samples-pos;if(n>320u)n=320u;int detected=0;
      double begin=micros(CLOCK_MONOTONIC);
      if(kws_engine_accept_pcm16(engine,pcm+pos,n,NULL,&detected)!=KWS_OK)return 2;
      double elapsed=micros(CLOCK_MONOTONIC)-begin;
      if(rep>0u){times[count++]=elapsed;wall_sum+=elapsed;hits+=(uint64_t)detected;}
    }
    if(rep>0u)cpu_us+=micros(CLOCK_PROCESS_CPUTIME_ID)-cpu_begin;
  }
  qsort(times,count,sizeof(*times),cmp);
  double audio=(double)samples/16000.0*(double)repeats;
  printf("{\"model_bytes\":%zu,\"keyword_pack_bytes\":%zu,\"arena_bytes\":%zu,\"alignment\":%zu,\"model_struct_bytes\":%zu,\"pack_struct_bytes\":%zu,\"input_buffer_bytes_20ms\":640,\"model_pack_read_wall_us\":%.3f,\"parse_validate_wall_us\":%.3f,\"arena_allocate_init_wall_us\":%.3f,\"startup_cache_condition\":\"warm-filesystem-no-cache-drop\",\"warmup_full_passes\":1,\"measured_repeats\":%u,\"audio_seconds_per_repeat\":%.6f,\"measured_audio_seconds\":%.6f,\"measured_blocks\":%zu,\"process_cpu_us\":%.3f,\"process_cpu_rtf\":%.9f,\"summed_block_wall_us\":%.3f,\"wall_rtf\":%.9f,\"p50_block_wall_us\":%.3f,\"p95_block_wall_us\":%.3f,\"p99_block_wall_us\":%.3f,\"max_block_wall_us\":%.3f,\"events\":%" PRIu64 "}\n",mn,kn,kws_engine_required_bytes(&model),kws_engine_required_alignment(),sizeof(model),sizeof(pack),read_us,parse_us,init_us,repeats,(double)samples/16000.0,audio,count,cpu_us,cpu_us/(audio*1000000.0),wall_sum,wall_sum/(audio*1000000.0),percentile(times,count,.5),percentile(times,count,.95),percentile(times,count,.99),times[count-1u],hits);
  free(times);free(pcm);free(arena);free(kb);free(mb);return 0;
}
