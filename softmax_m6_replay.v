`include "define.vh"

module softmax_m6_replay #(
    parameter integer ROW_LEN    = `SOFTMAX_ROW_LEN,
    parameter integer BLOCK_SIZE = `SOFTMAX_BLOCK_SIZE
) (
    input wire clk_i,
    input wire rst_i,
    input wire start_i,
    input wire [`SOFTMAX_M_W + `SOFTMAX_SUM_GLOBAL_W - 1:0] row_ctx_i,
    output reg ctrl_en_o,
    output reg [((ROW_LEN / BLOCK_SIZE) <= 1 ? 1 : $clog2(ROW_LEN / BLOCK_SIZE))-1:0] ctrl_addr_o,
    input wire [`SOFTMAX_M_W + BLOCK_SIZE-1:0] ctrl_rdata_i,
    output reg data_en_o,
    output reg [((ROW_LEN / BLOCK_SIZE) <= 1 ? 1 : $clog2(ROW_LEN / BLOCK_SIZE))-1:0] data_addr_o,
    input wire [BLOCK_SIZE*`SOFTMAX_DESC_W-1:0] data_rdata_i,
    output wire busy_o,
    output wire done_o,
    output reg out_valid_o,
    input wire out_ready_i,
    output reg [BLOCK_SIZE*`SOFTMAX_OUT_W-1:0] out_block_o,
    output reg out_last_o
);

  localparam integer BLOCK_COUNT = ROW_LEN / BLOCK_SIZE;
  localparam integer CTRL_ADDR_W = (BLOCK_COUNT <= 1) ? 1 : $clog2(BLOCK_COUNT);
  localparam integer DATA_ADDR_W = (BLOCK_COUNT <= 1) ? 1 : $clog2(BLOCK_COUNT);
  localparam integer DESC_FRAC_W = `SOFTMAX_DESC_FRAC_W;
  localparam integer EXP_W = `SOFTMAX_EXP_W;
  localparam integer DESC_SHIFT_FIELD_W = `SOFTMAX_DESC_W - `SOFTMAX_DESC_FRAC_W;
  localparam integer DECODE_SHIFT_W = (`SOFTMAX_DESC_W - `SOFTMAX_DESC_FRAC_W) + 1;
  localparam [1:0] ST_IDLE = 2'd0;
  localparam [1:0] ST_STREAM = 2'd1;
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

  function integer msb_pos;
    input [`SOFTMAX_SUM_GLOBAL_W-1:0] value_i;
    integer idx;
    begin
      msb_pos = 0;
      for (idx = `SOFTMAX_SUM_GLOBAL_W - 1; idx >= 0; idx = idx - 1) begin
        if (value_i[idx] && (msb_pos == 0)) begin
          msb_pos = idx;
        end
      end
    end
  endfunction

  reg [1:0] state_r;
  reg [$clog2(BLOCK_COUNT+1)-1:0] blk_rd_idx_r;
  reg signed [`SOFTMAX_M_W-1:0] m_global_final_r;
  reg signed [5:0] denom_e_r;
  reg [DESC_FRAC_W-1:0] denom_frac_q6_r;
  reg [DESC_FRAC_W-1:0] denom_comp_q6_r;

  reg [`SOFTMAX_SUM_GLOBAL_W-1:0] sum_global_raw;
  reg [`SOFTMAX_SUM_GLOBAL_W-1:0] norm_raw;
  integer norm_msb;
  integer lane_idx;
  reg signed [`SOFTMAX_M_W:0] gap_shift_w;
  reg signed [7:0] common_rshift_bias_w;
  reg signed [7:0] merged_exp_bias_w;
  reg signed [8:0] merged_exp_q6;
  reg signed [7:0] base_rshift;
  reg signed [7:0] final_rshift;
  reg [DECODE_SHIFT_W-1:0] shift_abs;
  reg [DESC_FRAC_W-1:0] frac_q6;
  reg [DESC_FRAC_W-1:0] lut_in_q6;
  reg [`SOFTMAX_DESC_W-1:0] desc_word;
  reg exp_rshift_extra;
  reg [EXP_W-1:0] exp_term;
  reg [`SOFTMAX_OUT_W-1:0] softmax_lane;
  wire signed [`SOFTMAX_M_W-1:0] m_local_w;
  wire [BLOCK_SIZE-1:0] mask_field_w;
  wire out_last_w;

  assign m_local_w    = ctrl_rdata_i[BLOCK_SIZE +: `SOFTMAX_M_W];
  assign mask_field_w = ctrl_rdata_i[BLOCK_SIZE-1:0];
  assign out_last_w   = (blk_rd_idx_r == BLOCK_COUNT-1);

  assign busy_o = (state_r != ST_IDLE);
  assign done_o = (state_r == ST_STREAM) && out_valid_o && out_ready_i && out_last_o;

  always @* begin
    ctrl_en_o   = 1'b0;
    ctrl_addr_o = {CTRL_ADDR_W{1'b0}};
    data_en_o   = 1'b0;
    data_addr_o = {DATA_ADDR_W{1'b0}};
    out_valid_o = 1'b0;
    out_block_o = {BLOCK_SIZE * `SOFTMAX_OUT_W{1'b0}};
    out_last_o  = 1'b0;

    case (state_r)
      ST_IDLE: begin
        if (start_i) begin
          ctrl_en_o   = 1'b1;
          ctrl_addr_o = {CTRL_ADDR_W{1'b0}};
          data_en_o   = 1'b1;
          data_addr_o = {DATA_ADDR_W{1'b0}};
        end
      end

      ST_STREAM: begin
        ctrl_en_o   = 1'b1;
        data_en_o   = 1'b1;
        out_valid_o = 1'b1;
        out_last_o  = out_last_w;
        if (out_ready_i && !out_last_w) begin
          ctrl_addr_o = blk_rd_idx_r[CTRL_ADDR_W-1:0] + 1'b1;
          data_addr_o = blk_rd_idx_r[DATA_ADDR_W-1:0] + 1'b1;
        end else begin
          ctrl_addr_o = blk_rd_idx_r[CTRL_ADDR_W-1:0];
          data_addr_o = blk_rd_idx_r[DATA_ADDR_W-1:0];
        end
        gap_shift_w = $signed({m_global_final_r[`SOFTMAX_M_W-1], m_global_final_r}) -
            $signed({m_local_w[`SOFTMAX_M_W-1], m_local_w});
        common_rshift_bias_w = gap_shift_w + $signed(denom_e_r);
        merged_exp_bias_w = -$signed({1'b0, denom_frac_q6_r}) - $signed({1'b0, denom_comp_q6_r});
        for (lane_idx = 0; lane_idx < BLOCK_SIZE; lane_idx = lane_idx + 1) begin
          softmax_lane = {`SOFTMAX_OUT_W{1'b0}};
          if (mask_field_w[lane_idx]) begin
            desc_word = data_rdata_i[lane_idx*`SOFTMAX_DESC_W+:`SOFTMAX_DESC_W];
            shift_abs = desc_shift_abs(desc_word);
            frac_q6 = desc_frac_raw(desc_word);
            base_rshift = common_rshift_bias_w + $signed({{(8 - DECODE_SHIFT_W) {1'b0}}, shift_abs});
            merged_exp_q6 = $signed({2'd0, frac_q6}) + merged_exp_bias_w;
            if (merged_exp_q6 < 0) begin
              lut_in_q6 = merged_exp_q6 + FRAC_UNIT;
              exp_rshift_extra = 1'b1;
            end else begin
              lut_in_q6 = merged_exp_q6[DESC_FRAC_W-1:0];
              exp_rshift_extra = 1'b0;
            end
            exp_term     = approx_exp_uq1_cfg(lut_in_q6);
            final_rshift = base_rshift + exp_rshift_extra;
            softmax_lane = (exp_term >> final_rshift) << `SOFTMAX_OUT_SHIFT;
          end
          out_block_o[lane_idx*`SOFTMAX_OUT_W+:`SOFTMAX_OUT_W] = softmax_lane;
        end
      end
    endcase
  end

  always @(posedge clk_i) begin
    if (rst_i) begin
      state_r          <= ST_IDLE;
      blk_rd_idx_r     <= {$clog2(BLOCK_COUNT + 1) {1'b0}};
      m_global_final_r <= {`SOFTMAX_M_W{1'b0}};
      denom_e_r        <= 6'sd0;
      denom_frac_q6_r  <= {DESC_FRAC_W{1'b0}};
      denom_comp_q6_r  <= {DESC_FRAC_W{1'b0}};
    end else begin
      case (state_r)
        ST_IDLE: begin
          if (start_i) begin
            m_global_final_r <= row_ctx_i[`SOFTMAX_SUM_GLOBAL_W+:`SOFTMAX_M_W];
            sum_global_raw = row_ctx_i[`SOFTMAX_SUM_GLOBAL_W-1:0];
            norm_msb       = msb_pos(sum_global_raw);
            if (sum_global_raw < (1 << `SOFTMAX_SUM_FRAC_W)) begin
              denom_e_r <= -6'sd1;
              norm_raw = sum_global_raw << 1;
            end else begin
              denom_e_r <= norm_msb - `SOFTMAX_SUM_FRAC_W;
              norm_raw = sum_global_raw >> (norm_msb - `SOFTMAX_SUM_FRAC_W);
            end
            denom_frac_q6_r <= norm_raw[DESC_FRAC_W-1:0];
            if (norm_raw[DESC_FRAC_W-1]) begin
              denom_comp_q6_r <= ((FRAC_UNIT - {1'b0, norm_raw[DESC_FRAC_W-1:0]}) >> 3) +
                                 ((FRAC_UNIT - {1'b0, norm_raw[DESC_FRAC_W-1:0]}) >> 4);
            end else begin
              denom_comp_q6_r <= (norm_raw[DESC_FRAC_W-1:0] >> 3) + (norm_raw[DESC_FRAC_W-1:0] >> 4);
            end
            blk_rd_idx_r <= {$clog2(BLOCK_COUNT + 1) {1'b0}};
            state_r      <= ST_STREAM;
          end
        end

        ST_STREAM: begin
          if (out_ready_i) begin
            if (out_last_w) begin
              state_r <= ST_IDLE;
            end else begin
              blk_rd_idx_r <= blk_rd_idx_r + 1'b1;
            end
          end
        end
      endcase
    end
  end

endmodule
