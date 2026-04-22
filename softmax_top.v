`include "define.vh"

module softmax_top #(
    parameter integer ROW_LEN    = `SOFTMAX_ROW_LEN,
    parameter integer BLOCK_SIZE = `SOFTMAX_BLOCK_SIZE
) (
    input wire clk_i,
    input wire rst_i,
    input wire in_valid_i,
    output wire in_ready_o,
    input wire [BLOCK_SIZE*`SOFTMAX_FP16_W-1:0] in_block_i,
    output wire out_valid_o,
    input wire out_ready_i,
    output wire [BLOCK_SIZE*`SOFTMAX_OUT_W-1:0] out_block_o,
    output wire out_last_o
);

  localparam integer BLOCK_COUNT = ROW_LEN / BLOCK_SIZE;
  localparam integer CTRL_ADDR_W = (BLOCK_COUNT <= 1) ? 1 : $clog2(BLOCK_COUNT);
  localparam integer DATA_ADDR_W = (BLOCK_COUNT <= 1) ? 1 : $clog2(BLOCK_COUNT);
  localparam integer WB_ENTRY_W = `SOFTMAX_M_W + BLOCK_SIZE + BLOCK_SIZE * `SOFTMAX_DESC_W;
  localparam integer CTRL_W = `SOFTMAX_M_W + BLOCK_SIZE;
  localparam integer ROW_CTX_W = `SOFTMAX_M_W + `SOFTMAX_SUM_GLOBAL_W;
  localparam integer SUM_LOCAL_W = $clog2(BLOCK_SIZE + 1) + `SOFTMAX_SUM_FRAC_W;

  wire        [   BLOCK_SIZE*`SOFTMAX_Y_W-1:0] y_block_w;
  wire signed [              `SOFTMAX_M_W-1:0] m_local_w;
  wire                                         prune_mode_w;
  wire                                         block_prune_w;
  wire        [                BLOCK_SIZE-1:0] mask_field_w;
  wire        [BLOCK_SIZE*`SOFTMAX_DESC_W-1:0] desc_block_w;
  wire                                         block_prune_m4_w;
  wire signed [              `SOFTMAX_M_W-1:0] m_local_m4_w;
  wire        [               SUM_LOCAL_W-1:0] sum_local_w;
  wire signed [              `SOFTMAX_M_W-1:0] m_global_w;
  wire                                         m_global_valid_w;
  wire signed [              `SOFTMAX_M_W-1:0] m_global_next_w;
  wire        [     `SOFTMAX_SUM_GLOBAL_W-1:0] sum_global_next_w;
  wire                                         row_done_w;

  reg                                          wr_row_active_r;
  reg                                          wr_slot_sel_r;
  reg         [                 CTRL_ADDR_W:0] wr_block_idx_r;
  reg         [                           1:0] slot_state_r          [0:1];
  reg                                          row_ctx_written_r     [0:1];
  reg                                          rd_slot_sel_r;

  wire        [                WB_ENTRY_W-1:0] wb_entry_w;

  wire                                         bw_idle_w;
  wire        [                 CTRL_ADDR_W:0] bw_blk_wr_count_w;
  wire                                         bw_row_start_w;
  wire                                         bw_enable_w;
  wire                                         bw_ctrl_en_w;
  wire                                         bw_ctrl_we_w;
  wire        [               CTRL_ADDR_W-1:0] bw_ctrl_addr_w;
  wire        [                    CTRL_W-1:0] bw_ctrl_wdata_w;
  wire                                         bw_data_en_w;
  wire                                         bw_data_we_w;
  wire        [               DATA_ADDR_W-1:0] bw_data_addr_w;
  wire        [BLOCK_SIZE*`SOFTMAX_DESC_W-1:0] bw_data_wdata_w;

  wire                                         m6_busy_w;
  wire                                         m6_done_w;
  wire                                         m6_start_w;
  wire                                         m6_start_slot_w;
  wire                                         m6_ctrl_en_w;
  wire        [               CTRL_ADDR_W-1:0] m6_ctrl_addr_w;
  wire                                         m6_data_en_w;
  wire        [               DATA_ADDR_W-1:0] m6_data_addr_w;
  wire        [                    CTRL_W-1:0] m6_ctrl_rdata_w;
  wire        [BLOCK_SIZE*`SOFTMAX_DESC_W-1:0] m6_data_rdata_w;
  wire        [                 ROW_CTX_W-1:0] m6_row_ctx_w;

  wire                                         slot0_ctrl_en_w;
  wire                                         slot0_ctrl_we_w;
  wire        [               CTRL_ADDR_W-1:0] slot0_ctrl_addr_w;
  wire        [                    CTRL_W-1:0] slot0_ctrl_wdata_w;
  wire        [                    CTRL_W-1:0] slot0_ctrl_rdata_w;
  wire                                         slot0_data_en_w;
  wire                                         slot0_data_we_w;
  wire        [               DATA_ADDR_W-1:0] slot0_data_addr_w;
  wire        [BLOCK_SIZE*`SOFTMAX_DESC_W-1:0] slot0_data_wdata_w;
  wire        [BLOCK_SIZE*`SOFTMAX_DESC_W-1:0] slot0_data_rdata_w;
  wire                                         slot0_row_ctx_we_w;
  wire        [                 ROW_CTX_W-1:0] slot0_row_ctx_wdata_w;
  wire        [                 ROW_CTX_W-1:0] slot0_row_ctx_rdata_w;

  wire                                         slot1_ctrl_en_w;
  wire                                         slot1_ctrl_we_w;
  wire        [               CTRL_ADDR_W-1:0] slot1_ctrl_addr_w;
  wire        [                    CTRL_W-1:0] slot1_ctrl_wdata_w;
  wire        [                    CTRL_W-1:0] slot1_ctrl_rdata_w;
  wire                                         slot1_data_en_w;
  wire                                         slot1_data_we_w;
  wire        [               DATA_ADDR_W-1:0] slot1_data_addr_w;
  wire        [BLOCK_SIZE*`SOFTMAX_DESC_W-1:0] slot1_data_wdata_w;
  wire        [BLOCK_SIZE*`SOFTMAX_DESC_W-1:0] slot1_data_rdata_w;
  wire                                         slot1_row_ctx_we_w;
  wire        [                 ROW_CTX_W-1:0] slot1_row_ctx_wdata_w;
  wire        [                 ROW_CTX_W-1:0] slot1_row_ctx_rdata_w;

  wire                                         any_write_busy_w;
  wire                                         any_idle_slot_w;
  wire                                         new_row_can_start_w;
  wire                                         accept_block_w;
  wire                                         row_start_w;
  wire                                         row_last_w;
  wire        [                           0:0] next_idle_slot_w;
  wire                                         slot0_write_done_w;
  wire                                         slot1_write_done_w;
  wire                                         slot0_m6_sel_w;
  wire                                         slot1_m6_sel_w;
  wire                                         slot0_bw_sel_w;
  wire                                         slot1_bw_sel_w;

  assign wb_entry_w = {m_local_w, mask_field_w, desc_block_w};

  assign any_write_busy_w = (slot_state_r[0] == `SOFTMAX_SLOT_WRITE_BUSY) || (slot_state_r[1] == `SOFTMAX_SLOT_WRITE_BUSY);
  assign any_idle_slot_w  = (slot_state_r[0] == `SOFTMAX_SLOT_IDLE) || (slot_state_r[1] == `SOFTMAX_SLOT_IDLE);
  assign next_idle_slot_w = (slot_state_r[0] == `SOFTMAX_SLOT_IDLE) ? 1'b0 : 1'b1;
  assign new_row_can_start_w = !wr_row_active_r && !any_write_busy_w && any_idle_slot_w;
  assign in_ready_o = wr_row_active_r ? 1'b1 : new_row_can_start_w;
  assign accept_block_w = in_valid_i && in_ready_o;
  assign row_start_w = accept_block_w && !wr_row_active_r;
  assign row_last_w = accept_block_w && (wr_block_idx_r == BLOCK_COUNT - 1);

  assign bw_row_start_w = row_start_w;
  assign bw_enable_w = any_write_busy_w || row_start_w;
  assign slot0_write_done_w = (slot_state_r[0] == `SOFTMAX_SLOT_WRITE_BUSY) && row_ctx_written_r[0] && bw_idle_w && (bw_blk_wr_count_w == BLOCK_COUNT);
  assign slot1_write_done_w = (slot_state_r[1] == `SOFTMAX_SLOT_WRITE_BUSY) && row_ctx_written_r[1] && bw_idle_w && (bw_blk_wr_count_w == BLOCK_COUNT);
  assign m6_start_w = !m6_busy_w && (slot0_write_done_w || slot1_write_done_w);
  assign m6_start_slot_w = slot0_write_done_w ? 1'b0 : 1'b1;
  assign m6_row_ctx_w = m6_start_w ? (m6_start_slot_w ? slot1_row_ctx_rdata_w : slot0_row_ctx_rdata_w)
                                   : (rd_slot_sel_r ? slot1_row_ctx_rdata_w : slot0_row_ctx_rdata_w);
  assign m6_ctrl_rdata_w = rd_slot_sel_r ? slot1_ctrl_rdata_w : slot0_ctrl_rdata_w;
  assign m6_data_rdata_w = rd_slot_sel_r ? slot1_data_rdata_w : slot0_data_rdata_w;

  assign slot0_m6_sel_w = (slot_state_r[0] == `SOFTMAX_SLOT_READ_BUSY) || (m6_start_w && (m6_start_slot_w == 1'b0));
  assign slot1_m6_sel_w = (slot_state_r[1] == `SOFTMAX_SLOT_READ_BUSY) || (m6_start_w && (m6_start_slot_w == 1'b1));
  assign slot0_bw_sel_w = ((slot_state_r[0] == `SOFTMAX_SLOT_WRITE_BUSY) ||
                           (row_start_w && (next_idle_slot_w == 1'b0))) && !slot0_m6_sel_w;
  assign slot1_bw_sel_w = ((slot_state_r[1] == `SOFTMAX_SLOT_WRITE_BUSY) ||
                           (row_start_w && (next_idle_slot_w == 1'b1))) && !slot1_m6_sel_w;

  assign slot0_ctrl_en_w = slot0_bw_sel_w ? bw_ctrl_en_w : slot0_m6_sel_w ? m6_ctrl_en_w : 1'b0;
  assign slot0_ctrl_we_w = slot0_bw_sel_w ? bw_ctrl_we_w : 1'b0;
  assign slot0_ctrl_addr_w = slot0_bw_sel_w ? bw_ctrl_addr_w : m6_ctrl_addr_w;
  assign slot0_ctrl_wdata_w = bw_ctrl_wdata_w;
  assign slot0_data_en_w = slot0_bw_sel_w ? bw_data_en_w : slot0_m6_sel_w ? m6_data_en_w : 1'b0;
  assign slot0_data_we_w = slot0_bw_sel_w ? bw_data_we_w : 1'b0;
  assign slot0_data_addr_w = slot0_bw_sel_w ? bw_data_addr_w : m6_data_addr_w;
  assign slot0_data_wdata_w = bw_data_wdata_w;

  assign slot1_ctrl_en_w = slot1_bw_sel_w ? bw_ctrl_en_w : slot1_m6_sel_w ? m6_ctrl_en_w : 1'b0;
  assign slot1_ctrl_we_w = slot1_bw_sel_w ? bw_ctrl_we_w : 1'b0;
  assign slot1_ctrl_addr_w = slot1_bw_sel_w ? bw_ctrl_addr_w : m6_ctrl_addr_w;
  assign slot1_ctrl_wdata_w = bw_ctrl_wdata_w;
  assign slot1_data_en_w = slot1_bw_sel_w ? bw_data_en_w : slot1_m6_sel_w ? m6_data_en_w : 1'b0;
  assign slot1_data_we_w = slot1_bw_sel_w ? bw_data_we_w : 1'b0;
  assign slot1_data_addr_w = slot1_bw_sel_w ? bw_data_addr_w : m6_data_addr_w;
  assign slot1_data_wdata_w = bw_data_wdata_w;

  assign slot0_row_ctx_we_w = accept_block_w && row_last_w && (wr_slot_sel_r == 1'b0);
  assign slot1_row_ctx_we_w = accept_block_w && row_last_w && (wr_slot_sel_r == 1'b1);
  assign slot0_row_ctx_wdata_w = {m_global_next_w, sum_global_next_w};
  assign slot1_row_ctx_wdata_w = {m_global_next_w, sum_global_next_w};

  softmax_m1_preprocess #(
      .BLOCK_SIZE(BLOCK_SIZE)
  ) u_m1 (
      .fp16_block_i(in_block_i),
      .y_block_o(y_block_w)
  );

  softmax_m2_block_ctx #(
      .BLOCK_SIZE(BLOCK_SIZE)
  ) u_m2 (
      .y_block_i(y_block_w),
      .m_global_i(m_global_w),
      .m_global_valid_i(m_global_valid_w),
      .m_local_o(m_local_w),
      .prune_mode_o(prune_mode_w),
      .block_prune_o(block_prune_w)
  );

  softmax_m3_desc_gen #(
      .BLOCK_SIZE(BLOCK_SIZE)
  ) u_m3 (
      .y_block_i(y_block_w),
      .m_local_i(m_local_w),
      .prune_mode_i(prune_mode_w),
      .block_prune_i(block_prune_w),
      .mask_field_o(mask_field_w),
      .desc_block_o(desc_block_w)
  );

  softmax_m4_block_sum #(
      .BLOCK_SIZE(BLOCK_SIZE)
  ) u_m4 (
      .block_prune_i(block_prune_w),
      .m_local_i(m_local_w),
      .mask_field_i(mask_field_w),
      .desc_block_i(desc_block_w),
      .block_prune_o(block_prune_m4_w),
      .m_local_o(m_local_m4_w),
      .sum_local_o(sum_local_w)
  );

  softmax_m5_online_update #(
      .BLOCK_SIZE(BLOCK_SIZE)
  ) u_m5 (
      .clk_i(clk_i),
      .rst_i(rst_i),
      .block_fire_i(accept_block_w),
      .row_start_i(row_start_w),
      .row_last_i(row_last_w),
      .block_prune_i(block_prune_m4_w),
      .m_local_i(m_local_m4_w),
      .sum_local_i(sum_local_w),
      .m_global_o(m_global_w),
      .m_global_valid_o(m_global_valid_w),
      .m_global_next_o(m_global_next_w),
      .sum_global_next_o(sum_global_next_w),
      .row_done_o(row_done_w)
  );

  softmax_bank_writer #(
      .ROW_LEN(ROW_LEN),
      .BLOCK_SIZE(BLOCK_SIZE)
  ) u_bank_writer (
      .clk_i(clk_i),
      .rst_i(rst_i),
      .row_start_i(bw_row_start_w),
      .enable_i(bw_enable_w),
      .entry_valid_i(accept_block_w),
      .entry_i(wb_entry_w),
      .ctrl_en_o(bw_ctrl_en_w),
      .ctrl_we_o(bw_ctrl_we_w),
      .ctrl_addr_o(bw_ctrl_addr_w),
      .ctrl_wdata_o(bw_ctrl_wdata_w),
      .data_en_o(bw_data_en_w),
      .data_we_o(bw_data_we_w),
      .data_addr_o(bw_data_addr_w),
      .data_wdata_o(bw_data_wdata_w),
      .idle_o(bw_idle_w),
      .blk_wr_count_o(bw_blk_wr_count_w)
  );

  softmax_row_slot #(
      .ROW_LEN(ROW_LEN),
      .BLOCK_SIZE(BLOCK_SIZE)
  ) u_slot0 (
      .clk_i(clk_i),
      .rst_i(rst_i),
      .ctrl_en_i(slot0_ctrl_en_w),
      .ctrl_we_i(slot0_ctrl_we_w),
      .ctrl_addr_i(slot0_ctrl_addr_w),
      .ctrl_wdata_i(slot0_ctrl_wdata_w),
      .ctrl_rdata_o(slot0_ctrl_rdata_w),
      .data_en_i(slot0_data_en_w),
      .data_we_i(slot0_data_we_w),
      .data_addr_i(slot0_data_addr_w),
      .data_wdata_i(slot0_data_wdata_w),
      .data_rdata_o(slot0_data_rdata_w),
      .row_ctx_we_i(slot0_row_ctx_we_w),
      .row_ctx_wdata_i(slot0_row_ctx_wdata_w),
      .row_ctx_rdata_o(slot0_row_ctx_rdata_w)
  );

  softmax_row_slot #(
      .ROW_LEN(ROW_LEN),
      .BLOCK_SIZE(BLOCK_SIZE)
  ) u_slot1 (
      .clk_i(clk_i),
      .rst_i(rst_i),
      .ctrl_en_i(slot1_ctrl_en_w),
      .ctrl_we_i(slot1_ctrl_we_w),
      .ctrl_addr_i(slot1_ctrl_addr_w),
      .ctrl_wdata_i(slot1_ctrl_wdata_w),
      .ctrl_rdata_o(slot1_ctrl_rdata_w),
      .data_en_i(slot1_data_en_w),
      .data_we_i(slot1_data_we_w),
      .data_addr_i(slot1_data_addr_w),
      .data_wdata_i(slot1_data_wdata_w),
      .data_rdata_o(slot1_data_rdata_w),
      .row_ctx_we_i(slot1_row_ctx_we_w),
      .row_ctx_wdata_i(slot1_row_ctx_wdata_w),
      .row_ctx_rdata_o(slot1_row_ctx_rdata_w)
  );

  softmax_m6_replay #(
      .ROW_LEN(ROW_LEN),
      .BLOCK_SIZE(BLOCK_SIZE)
  ) u_m6 (
      .clk_i(clk_i),
      .rst_i(rst_i),
      .start_i(m6_start_w),
      .row_ctx_i(m6_row_ctx_w),
      .ctrl_en_o(m6_ctrl_en_w),
      .ctrl_addr_o(m6_ctrl_addr_w),
      .ctrl_rdata_i(m6_ctrl_rdata_w),
      .data_en_o(m6_data_en_w),
      .data_addr_o(m6_data_addr_w),
      .data_rdata_i(m6_data_rdata_w),
      .busy_o(m6_busy_w),
      .done_o(m6_done_w),
      .out_valid_o(out_valid_o),
      .out_ready_i(out_ready_i),
      .out_block_o(out_block_o),
      .out_last_o(out_last_o)
  );

  integer slot_idx;
  always @(posedge clk_i) begin
    if (rst_i) begin
      wr_row_active_r <= 1'b0;
      wr_slot_sel_r   <= 1'b0;
      wr_block_idx_r  <= {(CTRL_ADDR_W + 1) {1'b0}};
      rd_slot_sel_r   <= 1'b0;
      for (slot_idx = 0; slot_idx < 2; slot_idx = slot_idx + 1) begin
        slot_state_r[slot_idx] <= `SOFTMAX_SLOT_IDLE;
        row_ctx_written_r[slot_idx] <= 1'b0;
      end
    end else begin
      if (accept_block_w) begin
        if (!wr_row_active_r) begin
          wr_row_active_r <= !row_last_w;
          wr_slot_sel_r <= next_idle_slot_w;
          wr_block_idx_r <= row_last_w ? {(CTRL_ADDR_W + 1) {1'b0}} : {{CTRL_ADDR_W{1'b0}}, 1'b1};
          slot_state_r[next_idle_slot_w] <= `SOFTMAX_SLOT_WRITE_BUSY;
          row_ctx_written_r[next_idle_slot_w] <= row_last_w;
        end else begin
          wr_row_active_r <= !row_last_w;
          wr_block_idx_r  <= row_last_w ? {(CTRL_ADDR_W + 1) {1'b0}} : (wr_block_idx_r + 1'b1);
          if (row_last_w) begin
            row_ctx_written_r[wr_slot_sel_r] <= 1'b1;
          end
        end
      end

      if (m6_start_w) begin
        slot_state_r[m6_start_slot_w] <= `SOFTMAX_SLOT_READ_BUSY;
        rd_slot_sel_r <= m6_start_slot_w;
      end

      if (m6_done_w) begin
        slot_state_r[rd_slot_sel_r] <= `SOFTMAX_SLOT_IDLE;
        row_ctx_written_r[rd_slot_sel_r] <= 1'b0;
      end
    end
  end

endmodule
