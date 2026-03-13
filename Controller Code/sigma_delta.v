module sigma_delta#(
    parameter STREAM_LENGTH = 16
)(
    input  wire                             clk,
    input  wire                             rst,
    input  wire                             en,
    input  wire signed [STREAM_LENGTH-1:0]  pcm_signal,
    output reg                              pdm_out
);

    // Stabiliy for pcm_signal (with overshoot)

  //  wire signed [STREAM_LENGTH-1:0] scaled_pcm = pcm_signal >>> 2;

    // Feedback logic - using 2 bit overshoot

    localparam signed [STREAM_LENGTH:0] FB_VAL = 17'sd32767; 
    wire signed [STREAM_LENGTH:0] fb = pdm_out ? FB_VAL : -FB_VAL;
    wire signed [STREAM_LENGTH:0] pcm_ext = $signed(pcm_signal);
    // Integrator Logic

    reg signed [23:0] integrator_1;
    reg signed [23:0] integrator_2;

    always @(posedge clk) begin
        if(rst) begin
            integrator_1 <= 24'd0;
            integrator_2 <= 24'd0;
            pdm_out      <= 1'b0;
        end
        else if(en) begin
            integrator_1 <= integrator_1 + (pcm_ext - fb);
            integrator_2 <= integrator_2 + (integrator_1-fb);
            pdm_out <= (integrator_2 >= 0);
        end
    end
endmodule
