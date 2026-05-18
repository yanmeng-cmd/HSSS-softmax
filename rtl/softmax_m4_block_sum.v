`include "define.vh"

module softmax_m4_block_sum #(
    parameter integer BLOCK_SIZE = `SOFTMAX_BLOCK_SIZE
) (
    input  wire                                                       block_prune_i,
    input  wire signed [                            `SOFTMAX_M_W-1:0] m_local_i,
    input  wire        [                              BLOCK_SIZE-1:0] mask_field_i,
    input  wire        [              BLOCK_SIZE*`SOFTMAX_DESC_W-1:0] desc_block_i,
    output reg                                                        block_prune_o,
    output reg signed  [                            `SOFTMAX_M_W-1:0] m_local_o,
    output reg         [$clog2(BLOCK_SIZE+1)+`SOFTMAX_SUM_FRAC_W-1:0] sum_local_o
);

  localparam integer SUM_LOCAL_W = $clog2(BLOCK_SIZE + 1) + `SOFTMAX_SUM_FRAC_W;
  localparam integer DESC_FRAC_W = `SOFTMAX_DESC_FRAC_W;
  localparam integer EXP_W = `SOFTMAX_EXP_W;
  localparam integer DESC_SHIFT_FIELD_W = `SOFTMAX_DESC_W - `SOFTMAX_DESC_FRAC_W;
  localparam integer DECODE_SHIFT_W = (`SOFTMAX_DESC_W - `SOFTMAX_DESC_FRAC_W) + 1;
  localparam integer PAIR_COUNT = (BLOCK_SIZE + 1) / 2;
  localparam integer QUAD_COUNT = (PAIR_COUNT + 1) / 2;
  localparam integer OCT_COUNT = (QUAD_COUNT + 1) / 2;
  localparam integer HEX_COUNT = (OCT_COUNT + 1) / 2;
  localparam integer FINAL_COUNT = (HEX_COUNT + 1) / 2;
  localparam [EXP_W-1:0] FRAC_UNIT = 1 << DESC_FRAC_W;

  function [EXP_W-1:0] approx_exp_uq1_cfg;
    input [DESC_FRAC_W-1:0] frac_i;
    reg [EXP_W-1:0] tri_term;
    reg [EXP_W-1:0] corr_term;
    begin
      if (frac_i[DESC_FRAC_W-1]) begin
        tri_term = FRAC_UNIT - {1'b0, frac_i};
      end else begin
        tri_term = {1'b0, frac_i};
      end
      corr_term = (tri_term >> 3) + (tri_term >> 4);
      approx_exp_uq1_cfg = FRAC_UNIT + {1'b0, frac_i} - corr_term;
    end
  endfunction

  function [DECODE_SHIFT_W-1:0] desc_shift_abs;
    input [`SOFTMAX_DESC_W-1:0] desc_i;
    reg [DESC_FRAC_W-1:0] low_bits_i;
    begin
      low_bits_i = desc_i[DESC_FRAC_W-1:0];
      desc_shift_abs =
          {{(DECODE_SHIFT_W - DESC_SHIFT_FIELD_W) {1'b0}}, desc_i[`SOFTMAX_DESC_W-1:DESC_FRAC_W]} +
          {{(DECODE_SHIFT_W - 1) {1'b0}}, |low_bits_i};
    end
  endfunction

  function [DESC_FRAC_W-1:0] desc_frac_raw;
    input [`SOFTMAX_DESC_W-1:0] desc_i;
    begin
      desc_frac_raw = {DESC_FRAC_W{1'b0}} - desc_i[DESC_FRAC_W-1:0];
    end
  endfunction

  function [EXP_W-1:0] desc_to_exp_uq1_cfg;
    input [`SOFTMAX_DESC_W-1:0] desc_i;
    reg [DECODE_SHIFT_W-1:0] shift_abs_i;
    reg [DESC_FRAC_W-1:0] frac_i;
    begin
      shift_abs_i = desc_shift_abs(desc_i);
      frac_i = desc_frac_raw(desc_i);
      desc_to_exp_uq1_cfg = approx_exp_uq1_cfg(frac_i) >> shift_abs_i;
    end
  endfunction

  integer lane_idx;
  integer pair_idx;
  integer quad_idx;
  integer oct_idx;
  integer hex_idx;
  integer final_idx;
  reg [EXP_W-1:0] lane_exp_r[0:BLOCK_SIZE-1];
  reg [SUM_LOCAL_W-1:0] pair_sum_r[0:PAIR_COUNT-1];
  reg [SUM_LOCAL_W-1:0] quad_sum_r[0:QUAD_COUNT-1];
  reg [SUM_LOCAL_W-1:0] oct_sum_r[0:OCT_COUNT-1];
  reg [SUM_LOCAL_W-1:0] hex_sum_r[0:HEX_COUNT-1];
  reg [SUM_LOCAL_W-1:0] final_sum_r[0:FINAL_COUNT-1];

  always @* begin
    block_prune_o = block_prune_i;
    m_local_o     = m_local_i;
    sum_local_o   = {SUM_LOCAL_W{1'b0}};
    for (lane_idx = 0; lane_idx < BLOCK_SIZE; lane_idx = lane_idx + 1) begin
      lane_exp_r[lane_idx] = {EXP_W{1'b0}};
    end
    for (pair_idx = 0; pair_idx < PAIR_COUNT; pair_idx = pair_idx + 1) begin
      pair_sum_r[pair_idx] = {SUM_LOCAL_W{1'b0}};
    end
    for (quad_idx = 0; quad_idx < QUAD_COUNT; quad_idx = quad_idx + 1) begin
      quad_sum_r[quad_idx] = {SUM_LOCAL_W{1'b0}};
    end
    for (oct_idx = 0; oct_idx < OCT_COUNT; oct_idx = oct_idx + 1) begin
      oct_sum_r[oct_idx] = {SUM_LOCAL_W{1'b0}};
    end
    for (hex_idx = 0; hex_idx < HEX_COUNT; hex_idx = hex_idx + 1) begin
      hex_sum_r[hex_idx] = {SUM_LOCAL_W{1'b0}};
    end
    for (final_idx = 0; final_idx < FINAL_COUNT; final_idx = final_idx + 1) begin
      final_sum_r[final_idx] = {SUM_LOCAL_W{1'b0}};
    end

    if (!block_prune_i) begin
      for (lane_idx = 0; lane_idx < BLOCK_SIZE; lane_idx = lane_idx + 1) begin
        if (mask_field_i[lane_idx]) begin
          lane_exp_r[lane_idx] =
              desc_to_exp_uq1_cfg(desc_block_i[lane_idx*`SOFTMAX_DESC_W+:`SOFTMAX_DESC_W]);
        end
      end

      for (pair_idx = 0; pair_idx < PAIR_COUNT; pair_idx = pair_idx + 1) begin
        pair_sum_r[pair_idx] = {{(SUM_LOCAL_W - EXP_W) {1'b0}}, lane_exp_r[2*pair_idx]};
        if ((2 * pair_idx + 1) < BLOCK_SIZE) begin
          pair_sum_r[pair_idx] = pair_sum_r[pair_idx] +
                                 {{(SUM_LOCAL_W-EXP_W){1'b0}}, lane_exp_r[2*pair_idx + 1]};
        end
      end

      for (quad_idx = 0; quad_idx < QUAD_COUNT; quad_idx = quad_idx + 1) begin
        quad_sum_r[quad_idx] = pair_sum_r[2*quad_idx];
        if ((2 * quad_idx + 1) < PAIR_COUNT) begin
          quad_sum_r[quad_idx] = quad_sum_r[quad_idx] + pair_sum_r[2*quad_idx+1];
        end
      end

      for (oct_idx = 0; oct_idx < OCT_COUNT; oct_idx = oct_idx + 1) begin
        oct_sum_r[oct_idx] = quad_sum_r[2*oct_idx];
        if ((2 * oct_idx + 1) < QUAD_COUNT) begin
          oct_sum_r[oct_idx] = oct_sum_r[oct_idx] + quad_sum_r[2*oct_idx+1];
        end
      end

      for (hex_idx = 0; hex_idx < HEX_COUNT; hex_idx = hex_idx + 1) begin
        hex_sum_r[hex_idx] = oct_sum_r[2*hex_idx];
        if ((2 * hex_idx + 1) < OCT_COUNT) begin
          hex_sum_r[hex_idx] = hex_sum_r[hex_idx] + oct_sum_r[2*hex_idx+1];
        end
      end

      for (final_idx = 0; final_idx < FINAL_COUNT; final_idx = final_idx + 1) begin
        final_sum_r[final_idx] = hex_sum_r[2*final_idx];
        if ((2 * final_idx + 1) < HEX_COUNT) begin
          final_sum_r[final_idx] = final_sum_r[final_idx] + hex_sum_r[2*final_idx+1];
        end
      end

      sum_local_o = final_sum_r[0];
    end
  end

endmodule
