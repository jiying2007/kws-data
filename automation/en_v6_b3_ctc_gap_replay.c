/* B3 research-only CTC policy replay over saved four-logit C11 frames.
 * No PCM/model/threshold changes; uses exact unmodified mero_ctc.c.
 * Usage: en_v6_b3_gap_replay logits.f32 max_token_gap_samples out.csv
 * Output: event_availability_sample,hey_sample,meh_sample.
 * This is a diagnostic trace, NOT a calibrated product decoder.
 */
#include "mero_ctc.h"
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <math.h>

static int parse_gap(const char *text,uint64_t *out) {
    char *end=NULL;
    unsigned long long n;
    if (!text || !*text || text[0]=='-' || !out) return -1;
    errno=0;
    n=strtoull(text,&end,10);
    if (errno!=0 || !end || *end!='\0' || n==0u ||
        (uint64_t)n>UINT64_MAX/2u) return -1;
    *out=(uint64_t)n;
    return 0;
}
static unsigned argmax(const float frame[4]) {
    unsigned idx=0u;
    for (unsigned i=1u;i<4u;++i)
        if (frame[i]>frame[idx]) idx=i;
    return idx;
}
int main(int argc,char **argv) {
    FILE *input=NULL,*output=NULL;
    en_v5_mero_state state;
    uint64_t gap=0u,frames=0u,events=0u;
    uint64_t hey_clock=0u,meh_clock=0u;
    int rc=1;
    if (argc!=4 || parse_gap(argv[2],&gap)!=0) {
        fputs("Usage: gap_replay logits.f32 positive_max_gap_samples out.csv\n",
              stderr);
        return 2;
    }
    if (en_v5_mero_init(&state,gap,0u)!=EN_V5_OK)
        return 3;
    input=fopen(argv[1],"rb");
    if (!input) {perror("logits input");return 4;}
    output=fopen(argv[3],"w");
    if (!output) {perror("event output");fclose(input);return 5;}
    for (;;) {
        float frame[4];
        size_t received=fread(frame,sizeof(float),4u,input);
        if (received==0u && !ferror(input)) break;
        if (received!=4u || frames>100000u ||
            frames>(UINT64_MAX-512u)/160u) {
            fputs("STOP: logits shape/length/read error\n",stderr);
            goto finish;
        }
        uint64_t clock=512u+160u*frames;
        unsigned token=argmax(frame);
        unsigned before_argmax=state.previous_argmax;
        en_v5_mero_event ev={0};
        if (en_v5_mero_accept(&state,frame,clock,&ev)!=EN_V5_OK) {
            fputs("STOP: invalid, nonfinite or nonmonotone logits\n",stderr);
            goto finish;
        }
        if (token!=before_argmax && token==1u && state.progress==1u) {
            hey_clock=clock;meh_clock=0u;
        } else if (token!=before_argmax && token==2u &&
                   state.progress==2u) {
            meh_clock=clock;
        }
        if (ev.detected) {
            if (!(hey_clock>0u && meh_clock>hey_clock &&
                  clock>meh_clock && meh_clock-hey_clock<=gap &&
                  clock-meh_clock<=gap)) {
                fputs("STOP: unproven ordered event anchors\n",stderr);
                goto finish;
            }
            if (fprintf(output,"%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                        clock,hey_clock,meh_clock)<0) {
                fputs("STOP: event output write\n",stderr);
                goto finish;
            }
            ++events;
            hey_clock=0u;meh_clock=0u;
        }
        ++frames;
    }
    if (frames==0u || ferror(input) || fflush(output)!=0 ||
        ferror(output)) {
        fputs("STOP: no frames or write error\n",stderr);
        goto finish;
    }
    rc=0;
    fprintf(stderr,"B3_GAP_CTC_REPLAY_OK gap=%" PRIu64
            " frames=%" PRIu64 " events=%" PRIu64 "\n",
            gap,frames,events);
finish:
    if (input && fclose(input)!=0) rc=1;
    if (output && fclose(output)!=0) rc=1;
    return rc;
}
