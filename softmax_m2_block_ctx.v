`include "define.vh"

module softmax_m2_block_ctx #(
    parameter integer BLOCK_SIZE = `SOFTMAX_BLOCK_SIZE
) (
    input  wire        [BLOCK_SIZE*`SOFTMAX_Y_W-1:0] y_block_i,
    input  wire signed [           `SOFTMAX_M_W-1:0] m_global_i,
    input  wire                                      m_global_valid_i,
    output reg signed  [           `SOFTMAX_M_W-1:0] m_local_o,
    output reg                                       prune_mode_o,
    output reg                                       block_prune_o
);

  localparam integer BLOCK_LOG2 = $clog2(BLOCK_SIZE);
  localparam integer SUM_FRAC_W = `SOFTMAX_SUM_FRAC_W;
  localparam integer NEAR_PEAK_CNT_W = $clog2(BLOCK_SIZE + 1);
  localparam integer DELTA_W = `SOFTMAX_M_W + 1;
  localparam integer PAIR_COUNT = (BLOCK_SIZE + 1) / 2;
  localparam integer QUAD_COUNT = (PAIR_COUNT + 1) / 2;
  localparam integer OCT_COUNT = (QUAD_COUNT + 1) / 2;
  localparam integer HEX_COUNT = (OCT_COUNT + 1) / 2;
  localparam integer FINAL_COUNT = (HEX_COUNT + 1) / 2;
  localparam [NEAR_PEAK_CNT_W-1:0] NEAR_PEAK_CNT_TH = (BLOCK_SIZE - 1) / 2;
  localparam [NEAR_PEAK_CNT_W-1:0] NEAR_PEAK_LIMIT = NEAR_PEAK_CNT_TH + 1'b1;
  localparam signed [DELTA_W-1:0] TAU_BLK_BASE = -(SUM_FRAC_W + BLOCK_LOG2);

  function signed [`SOFTMAX_Y_W-1:0] max2_signed_y;
    input signed [`SOFTMAX_Y_W-1:0] lhs_i;
    input signed [`SOFTMAX_Y_W-1:0] rhs_i;
    begin
      max2_signed_y = (lhs_i >= rhs_i) ? lhs_i : rhs_i;
    end
  endfunction

  function [NEAR_PEAK_CNT_W-1:0] sat_add_hits;
    input [NEAR_PEAK_CNT_W-1:0] lhs_i;
    input [NEAR_PEAK_CNT_W-1:0] rhs_i;
    reg [NEAR_PEAK_CNT_W:0] sum_raw;
    begin
      sum_raw = lhs_i + rhs_i;
      if (sum_raw >= {1'b0, NEAR_PEAK_LIMIT}) begin
        sat_add_hits = NEAR_PEAK_LIMIT;
      end else begin
        sat_add_hits = sum_raw[NEAR_PEAK_CNT_W-1:0];
      end
    end
  endfunction

  integer                          lane_idx;
  integer                          pair_idx;
  integer                          quad_idx;
  integer                          oct_idx;
  integer                          hex_idx;
  integer                          final_idx;

  reg signed [   `SOFTMAX_Y_W-1:0] lane_y_r           [ 0:BLOCK_SIZE-1];
  reg signed [   `SOFTMAX_Y_W-1:0] max_pair_r         [ 0:PAIR_COUNT-1];
  reg signed [   `SOFTMAX_Y_W-1:0] max_quad_r         [ 0:QUAD_COUNT-1];
  reg signed [   `SOFTMAX_Y_W-1:0] max_oct_r          [  0:OCT_COUNT-1];
  reg signed [   `SOFTMAX_Y_W-1:0] max_hex_r          [  0:HEX_COUNT-1];
  reg signed [   `SOFTMAX_Y_W-1:0] max_final_r        [0:FINAL_COUNT-1];

  reg                              near_peak_hit_r    [ 0:BLOCK_SIZE-1];
  reg        [NEAR_PEAK_CNT_W-1:0] hit_pair_r         [ 0:PAIR_COUNT-1];
  reg        [NEAR_PEAK_CNT_W-1:0] hit_quad_r         [ 0:QUAD_COUNT-1];
  reg        [NEAR_PEAK_CNT_W-1:0] hit_oct_r          [  0:OCT_COUNT-1];
  reg        [NEAR_PEAK_CNT_W-1:0] hit_hex_r          [  0:HEX_COUNT-1];
  reg        [NEAR_PEAK_CNT_W-1:0] hit_final_r        [0:FINAL_COUNT-1];

  reg signed [        DELTA_W-1:0] delta_blk;
  reg signed [        DELTA_W-1:0] tau_blk_sel;
  reg signed [        DELTA_W-1:0] m_local_ext;
  reg signed [        DELTA_W-1:0] m_global_ext;
  reg signed [   `SOFTMAX_Y_W-1:0] ymax_wide;
  reg signed [   `SOFTMAX_M_W-1:0] ymax_floor;
  reg                              frac_nonzero;
  reg signed [   `SOFTMAX_M_W-1:0] near_peak_floor_th;

  always @* begin
    for (lane_idx = 0; lane_idx < BLOCK_SIZE; lane_idx = lane_idx + 1) begin
      lane_y_r[lane_idx]        = $signed(y_block_i[lane_idx*`SOFTMAX_Y_W+:`SOFTMAX_Y_W]);
      near_peak_hit_r[lane_idx] = 1'b0;
    end
    for (pair_idx = 0; pair_idx < PAIR_COUNT; pair_idx = pair_idx + 1) begin
      max_pair_r[pair_idx] = {`SOFTMAX_Y_W{1'b0}};
      hit_pair_r[pair_idx] = {NEAR_PEAK_CNT_W{1'b0}};
    end
    for (quad_idx = 0; quad_idx < QUAD_COUNT; quad_idx = quad_idx + 1) begin
      max_quad_r[quad_idx] = {`SOFTMAX_Y_W{1'b0}};
      hit_quad_r[quad_idx] = {NEAR_PEAK_CNT_W{1'b0}};
    end
    for (oct_idx = 0; oct_idx < OCT_COUNT; oct_idx = oct_idx + 1) begin
      max_oct_r[oct_idx] = {`SOFTMAX_Y_W{1'b0}};
      hit_oct_r[oct_idx] = {NEAR_PEAK_CNT_W{1'b0}};
    end
    for (hex_idx = 0; hex_idx < HEX_COUNT; hex_idx = hex_idx + 1) begin
      max_hex_r[hex_idx] = {`SOFTMAX_Y_W{1'b0}};
      hit_hex_r[hex_idx] = {NEAR_PEAK_CNT_W{1'b0}};
    end
    for (final_idx = 0; final_idx < FINAL_COUNT; final_idx = final_idx + 1) begin
      max_final_r[final_idx] = {`SOFTMAX_Y_W{1'b0}};
      hit_final_r[final_idx] = {NEAR_PEAK_CNT_W{1'b0}};
    end

    for (pair_idx = 0; pair_idx < PAIR_COUNT; pair_idx = pair_idx + 1) begin
      max_pair_r[pair_idx] = lane_y_r[2*pair_idx];
      if ((2 * pair_idx + 1) < BLOCK_SIZE) begin
        max_pair_r[pair_idx] = max2_signed_y(max_pair_r[pair_idx], lane_y_r[2*pair_idx+1]);
      end
    end

    for (quad_idx = 0; quad_idx < QUAD_COUNT; quad_idx = quad_idx + 1) begin
      max_quad_r[quad_idx] = max_pair_r[2*quad_idx];
      if ((2 * quad_idx + 1) < PAIR_COUNT) begin
        max_quad_r[quad_idx] = max2_signed_y(max_quad_r[quad_idx], max_pair_r[2*quad_idx+1]);
      end
    end

    for (oct_idx = 0; oct_idx < OCT_COUNT; oct_idx = oct_idx + 1) begin
      max_oct_r[oct_idx] = max_quad_r[2*oct_idx];
      if ((2 * oct_idx + 1) < QUAD_COUNT) begin
        max_oct_r[oct_idx] = max2_signed_y(max_oct_r[oct_idx], max_quad_r[2*oct_idx+1]);
      end
    end

    for (hex_idx = 0; hex_idx < HEX_COUNT; hex_idx = hex_idx + 1) begin
      max_hex_r[hex_idx] = max_oct_r[2*hex_idx];
      if ((2 * hex_idx + 1) < OCT_COUNT) begin
        max_hex_r[hex_idx] = max2_signed_y(max_hex_r[hex_idx], max_oct_r[2*hex_idx+1]);
      end
    end

    for (final_idx = 0; final_idx < FINAL_COUNT; final_idx = final_idx + 1) begin
      max_final_r[final_idx] = max_hex_r[2*final_idx];
      if ((2 * final_idx + 1) < HEX_COUNT) begin
        max_final_r[final_idx] = max2_signed_y(max_final_r[final_idx], max_hex_r[2*final_idx+1]);
      end
    end

    ymax_wide = max_final_r[0];
    ymax_floor = $signed(ymax_wide) >>> `SOFTMAX_Y_FRAC_W;
    frac_nonzero = |ymax_wide[`SOFTMAX_Y_FRAC_W-1:0];
    m_local_o = ymax_floor + (frac_nonzero ? 7'sd1 : 7'sd0);
    near_peak_floor_th = m_local_o - 7'sd2;

    for (lane_idx = 0; lane_idx < BLOCK_SIZE; lane_idx = lane_idx + 1) begin
      near_peak_hit_r[lane_idx] = ($signed(lane_y_r[lane_idx]) >>> `SOFTMAX_Y_FRAC_W) >=
          near_peak_floor_th;
    end

    for (pair_idx = 0; pair_idx < PAIR_COUNT; pair_idx = pair_idx + 1) begin
      hit_pair_r[pair_idx] = {{(NEAR_PEAK_CNT_W - 1) {1'b0}}, near_peak_hit_r[2*pair_idx]};
      if ((2 * pair_idx + 1) < BLOCK_SIZE) begin
        hit_pair_r[pair_idx] = sat_add_hits(
            hit_pair_r[pair_idx], {{(NEAR_PEAK_CNT_W - 1) {1'b0}}, near_peak_hit_r[2*pair_idx+1]});
      end
    end

    for (quad_idx = 0; quad_idx < QUAD_COUNT; quad_idx = quad_idx + 1) begin
      hit_quad_r[quad_idx] = hit_pair_r[2*quad_idx];
      if ((2 * quad_idx + 1) < PAIR_COUNT) begin
        hit_quad_r[quad_idx] = sat_add_hits(hit_quad_r[quad_idx], hit_pair_r[2*quad_idx+1]);
      end
    end

    for (oct_idx = 0; oct_idx < OCT_COUNT; oct_idx = oct_idx + 1) begin
      hit_oct_r[oct_idx] = hit_quad_r[2*oct_idx];
      if ((2 * oct_idx + 1) < QUAD_COUNT) begin
        hit_oct_r[oct_idx] = sat_add_hits(hit_oct_r[oct_idx], hit_quad_r[2*oct_idx+1]);
      end
    end

    for (hex_idx = 0; hex_idx < HEX_COUNT; hex_idx = hex_idx + 1) begin
      hit_hex_r[hex_idx] = hit_oct_r[2*hex_idx];
      if ((2 * hex_idx + 1) < OCT_COUNT) begin
        hit_hex_r[hex_idx] = sat_add_hits(hit_hex_r[hex_idx], hit_oct_r[2*hex_idx+1]);
      end
    end

    for (final_idx = 0; final_idx < FINAL_COUNT; final_idx = final_idx + 1) begin
      hit_final_r[final_idx] = hit_hex_r[2*final_idx];
      if ((2 * final_idx + 1) < HEX_COUNT) begin
        hit_final_r[final_idx] = sat_add_hits(hit_final_r[final_idx], hit_hex_r[2*final_idx+1]);
      end
    end

    prune_mode_o = (hit_final_r[0] <= NEAR_PEAK_CNT_TH);
    tau_blk_sel  = TAU_BLK_BASE;
    m_local_ext  = {m_local_o[`SOFTMAX_M_W-1], m_local_o};
    m_global_ext = {m_global_i[`SOFTMAX_M_W-1], m_global_i};
    delta_blk    = m_local_ext - m_global_ext;
    block_prune_o = m_global_valid_i && (delta_blk <= tau_blk_sel);
  end

endmodule
