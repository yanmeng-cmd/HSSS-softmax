`include "define.vh"

module softmax_m3_desc_gen #(
    parameter integer BLOCK_SIZE = `SOFTMAX_BLOCK_SIZE
) (
    input  wire        [   BLOCK_SIZE*`SOFTMAX_Y_W-1:0] y_block_i,
    input  wire signed [              `SOFTMAX_M_W-1:0] m_local_i,
    input  wire                                         prune_mode_i,
    input  wire                                         block_prune_i,
    output reg         [                BLOCK_SIZE-1:0] mask_field_o,
    output reg         [BLOCK_SIZE*`SOFTMAX_DESC_W-1:0] desc_block_o
);

  localparam integer TAU_W = `SOFTMAX_M_W + 1;
  localparam integer Z_W = `SOFTMAX_Y_W + 1;
  localparam signed [TAU_W-1:0] TAU_ELEM_MODE1 = -2;
  localparam signed [TAU_W-1:0] TAU_ELEM_MODE0 = -4;

  integer lane_idx;
  reg signed [`SOFTMAX_Y_W-1:0] lane_y;
  reg signed [`SOFTMAX_M_W-1:0] lane_floor_int;
  reg signed [TAU_W-1:0] tau_elem;
  reg signed [TAU_W-1:0] keep_threshold;
  reg signed [TAU_W-1:0] lane_floor_ext;
  reg signed [Z_W-1:0] m_local_q10;
  reg signed [Z_W-1:0] z_wide;
  localparam signed [Z_W-1:0] DESC_CODE_MAX = (1 << `SOFTMAX_DESC_W) - 1;

  reg signed [Z_W-1:0] z_q6_raw;
  reg signed [Z_W-1:0] neg_z_q6_raw;

  always @* begin
    mask_field_o = {BLOCK_SIZE{1'b0}};
    desc_block_o = {BLOCK_SIZE * `SOFTMAX_DESC_W{1'b0}};
    tau_elem = prune_mode_i ? TAU_ELEM_MODE1 : TAU_ELEM_MODE0;
    keep_threshold = {m_local_i[`SOFTMAX_M_W-1], m_local_i} + tau_elem;
    m_local_q10 = $signed({{(Z_W - `SOFTMAX_M_W) {m_local_i[`SOFTMAX_M_W-1]}}, m_local_i}) <<<
        `SOFTMAX_Y_FRAC_W;

    if (!block_prune_i) begin
      for (lane_idx = 0; lane_idx < BLOCK_SIZE; lane_idx = lane_idx + 1) begin
        lane_y = $signed(y_block_i[lane_idx*`SOFTMAX_Y_W+:`SOFTMAX_Y_W]);
        lane_floor_int = $signed(lane_y) >>> `SOFTMAX_Y_FRAC_W;
        lane_floor_ext = {lane_floor_int[`SOFTMAX_M_W-1], lane_floor_int};
        if (lane_floor_ext >= keep_threshold) begin
          mask_field_o[lane_idx] = 1'b1;

          z_wide = $signed({lane_y[`SOFTMAX_Y_W-1], lane_y}) - m_local_q10;
          z_q6_raw = z_wide >>> (`SOFTMAX_Y_FRAC_W - `SOFTMAX_DESC_FRAC_W);

          if (z_q6_raw >= 0) begin
            desc_block_o[lane_idx*`SOFTMAX_DESC_W+:`SOFTMAX_DESC_W] = {`SOFTMAX_DESC_W{1'b0}};
          end else begin
            neg_z_q6_raw = -z_q6_raw;
            if (neg_z_q6_raw >= DESC_CODE_MAX) begin
              desc_block_o[lane_idx*`SOFTMAX_DESC_W+:`SOFTMAX_DESC_W] = DESC_CODE_MAX[`SOFTMAX_DESC_W-1:0];
            end else begin
              desc_block_o[lane_idx*`SOFTMAX_DESC_W+:`SOFTMAX_DESC_W] = neg_z_q6_raw[`SOFTMAX_DESC_W-1:0];
            end
          end
        end
      end
    end
  end

endmodule
