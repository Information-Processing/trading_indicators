#include <ap_fixed.h>
#include <ap_int.h>

typedef ap_fixed<18, 13> fixed_t;
typedef ap_fixed<64, 40> acc_t;
#define D 13
typedef ap_uint<512> wide_bus_t;

void outer_product_accum(
    wide_bus_t *mem_in,
    int num_samples,
    fixed_t AtA[D*D],
    fixed_t Atb[D]
)
{
#pragma HLS INTERFACE m_axi     port=mem_in  offset=slave bundle=gmem0 max_read_burst_length=256
#pragma HLS INTERFACE s_axilite port=mem_in  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=num_samples bundle=CTRL
#pragma HLS INTERFACE s_axilite port=AtA bundle=CTRL
#pragma HLS INTERFACE s_axilite port=Atb bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    acc_t acc_AtA[D][D];
    acc_t acc_Atb[D];

#pragma HLS ARRAY_PARTITION variable=acc_AtA complete dim=0
#pragma HLS ARRAY_PARTITION variable=acc_Atb complete

Init:
    for (int i = 0; i < D; i++) {
        #pragma HLS UNROLL
        acc_Atb[i] = 0;
        for (int j = 0; j < D; j++) {
            #pragma HLS UNROLL
            acc_AtA[i][j] = 0;
        }
    }

Sample_Loop:
    for (int s = 0; s < num_samples; s++) {
        #pragma HLS PIPELINE II=1

        wide_bus_t raw_data = mem_in[s];
        fixed_t x[D];
        #pragma HLS ARRAY_PARTITION variable=x complete
        fixed_t y;

        // Unpack Target from 32-bit aligned slot 0
        y.range() = raw_data.range(17, 0);

        // Unpack Features from 32-bit aligned slots 1..D
        for (int k = 0; k < D; k++) {
            #pragma HLS UNROLL
            x[k].range() = raw_data.range((k + 1) * 32 + 17, (k + 1) * 32);
        }

        // Update Vector
        for (int i = 0; i < D; i++) {
            #pragma HLS UNROLL
            #pragma HLS BIND_OP variable=acc_Atb op=add impl=dsp
            acc_Atb[i] += (acc_t)(x[i] * y);
        }

        // Update Full Matrix (Calculating all D*D cells ensures II=1)
        for (int i = 0; i < D; i++) {
            #pragma HLS UNROLL
            for (int j = 0; j < D; j++) {
                #pragma HLS UNROLL
                #pragma HLS BIND_OP variable=acc_AtA op=add impl=dsp
                acc_AtA[i][j] += (acc_t)(x[i] * x[j]);
            }
        }
    }

Flush:
    for (int i = 0; i < D; i++) {
        #pragma HLS UNROLL
        Atb[i] = (fixed_t)acc_Atb[i];
        for (int j = 0; j < D; j++) {
            #pragma HLS UNROLL
            AtA[i*D + j] = (fixed_t)acc_AtA[i][j];
        }
    }
}
