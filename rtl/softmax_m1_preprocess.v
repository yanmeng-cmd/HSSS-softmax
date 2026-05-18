`include "define.vh"

module softmax_m1_preprocess #(
    parameter integer BLOCK_SIZE = `SOFTMAX_BLOCK_SIZE
) (
    input  wire [BLOCK_SIZE*`SOFTMAX_FP16_W-1:0] fp16_block_i,
    output reg  [BLOCK_SIZE*`SOFTMAX_Y_W-1:0]    y_block_o
);

  function signed [`SOFTMAX_Y_W-1:0] sat_q6_10;
    input signed [31:0] value_i;
    begin
      if (value_i > 32'sd32767) begin
        sat_q6_10 = 16'sd32767;
      end else if (value_i < -32'sd32768) begin
        sat_q6_10 = -16'sd32768;
      end else begin
        sat_q6_10 = value_i[`SOFTMAX_Y_W-1:0];
      end
    end
  endfunction

  function signed [`SOFTMAX_Y_W-1:0] fp16_to_q6_10;
    input [`SOFTMAX_FP16_W-1:0] fp_i;
    reg sign_bit;
    reg [4:0] exp_field;
    reg [9:0] frac_field;
    reg signed [31:0] scaled_raw;
    integer exp_unbias;
    begin
      sign_bit   = fp_i[15];
      exp_field  = fp_i[14:10];
      frac_field = fp_i[9:0];

      if (exp_field == 5'd0) begin
        fp16_to_q6_10 = 16'sd0;
      end else if (exp_field == 5'h1f) begin
        fp16_to_q6_10 = sign_bit ? -16'sd32768 : 16'sd32767;
      end else begin
        exp_unbias = exp_field - 15;
        scaled_raw = $signed({21'd0, 1'b1, frac_field});
        if (exp_unbias >= 0) begin
          scaled_raw = scaled_raw <<< exp_unbias;
        end else begin
          scaled_raw = scaled_raw >>> (-exp_unbias);
        end
        if (sign_bit) begin
          scaled_raw = -scaled_raw;
        end
        fp16_to_q6_10 = sat_q6_10(scaled_raw);
      end
    end
  endfunction

  function signed [`SOFTMAX_Y_W-1:0] mul_log2e_23_div_16;
    input signed [`SOFTMAX_Y_W-1:0] x_i;
    reg signed [31:0] tmp_value;
    begin
      tmp_value = $signed(x_i) + ($signed(x_i) >>> 1) - ($signed(x_i) >>> 4);
      mul_log2e_23_div_16 = sat_q6_10(tmp_value);
    end
  endfunction

  integer lane_idx;
  reg [`SOFTMAX_FP16_W-1:0] fp16_lane;
  reg signed [`SOFTMAX_Y_W-1:0] x_q6_10;
  reg signed [`SOFTMAX_Y_W-1:0] y_q6_10;

  always @* begin
    y_block_o = {BLOCK_SIZE * `SOFTMAX_Y_W{1'b0}};
    for (lane_idx = 0; lane_idx < BLOCK_SIZE; lane_idx = lane_idx + 1) begin
      fp16_lane = fp16_block_i[lane_idx*`SOFTMAX_FP16_W+:`SOFTMAX_FP16_W];
      x_q6_10 = fp16_to_q6_10(fp16_lane);
      y_q6_10 = mul_log2e_23_div_16(x_q6_10);
      y_block_o[lane_idx*`SOFTMAX_Y_W+:`SOFTMAX_Y_W] = y_q6_10;
    end
  end

endmodule
