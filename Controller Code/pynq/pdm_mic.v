`timescale 1ns / 1ps

module pdm_microphone
    #(
         parameter INPUT_FREQ = 50_000_000,
         parameter PDM_FREQ = 2_400_000
     )
     (
         input         clk,
         input         rst,
         output [31:0] mic_pcm_data,
         output        mic_data_valid,
         output        mic_clk,
         input         mic_pdm_data,
         output        mic_lrsel
     );

    reg [2:0]      data_reg;
    wire            clk_rising;

    // Shift register to align incoming PDM data
    always @(posedge clk) begin
        if (rst) begin
            data_reg       <= 0;

        end
        else begin
            data_reg[0]   <= mic_pdm_data;
            data_reg[2:1] <= data_reg[1:0];

        end
    end

    // clock gen
    pdm_clk_gen
        #(
            .INPUT_FREQ(INPUT_FREQ),
            .OUTPUT_FREQ(PDM_FREQ)
        )
        pdm_clk_gen_i
        (
            .clk(clk),
            .rst(rst),
            .M_CLK(mic_clk),
            .m_clk_rising(clk_rising)
        );

    wire [31:0] cic_out_data;
    wire        cic_out_valid;

    cic_compiler_0 cic_compiler
                   (
                       .aclk(clk),
                       .s_axis_data_tdata({1'b0,data_reg[2]}),
                       .s_axis_data_tvalid(clk_rising),
                       .s_axis_data_tready(),
                       .m_axis_data_tdata(cic_out_data),
                       .m_axis_data_tvalid(1'b1)
                   );

    wire [31:0] fir_out_data;
    wire        fir_out_valid;

        fir_compiler_0 fir_compiler
                   (
                       .aclk(clk),
                       .s_axis_data_tdata(cic_out_data),
                       .s_axis_data_tvalid(cic_out_valid),
                       .s_axis_data_tready(),
                       .m_axis_data_tdata(fir_out_data),
                       .m_axis_data_tvalid(1'b1)
                   );


    assign mic_lrsel = 0;
    assign mic_pcm_data = fir_out_data;
    assign mic_data_valid = fir_out_valid;

endmodule
