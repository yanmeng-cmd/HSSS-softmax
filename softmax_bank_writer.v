`include "define.vh"

module softmax_bank_writer #(
    parameter integer ROW_LEN    = `SOFTMAX_ROW_LEN,
    parameter integer BLOCK_SIZE = `SOFTMAX_BLOCK_SIZE
) (
    input wire clk_i,
    input wire rst_i,
    input wire row_start_i,
    input wire enable_i,
    input wire entry_valid_i,
    input wire [`SOFTMAX_M_W + BLOCK_SIZE + BLOCK_SIZE*`SOFTMAX_DESC_W - 1:0] entry_i,
    output reg ctrl_en_o,
    output reg ctrl_we_o,
    output reg [((ROW_LEN / BLOCK_SIZE) <= 1 ? 1 : $clog2(ROW_LEN / BLOCK_SIZE))-1:0] ctrl_addr_o,
    output reg [`SOFTMAX_M_W + BLOCK_SIZE-1:0] ctrl_wdata_o,
    output reg data_en_o,
    output reg data_we_o,
    output reg [((ROW_LEN / BLOCK_SIZE) <= 1 ? 1 : $clog2(ROW_LEN / BLOCK_SIZE))-1:0] data_addr_o,
    output reg [BLOCK_SIZE*`SOFTMAX_DESC_W-1:0] data_wdata_o,
    output wire idle_o,
    output wire [$clog2((ROW_LEN / BLOCK_SIZE)+1)-1:0] blk_wr_count_o
);

  localparam integer BLOCK_COUNT = ROW_LEN / BLOCK_SIZE;
  localparam integer CTRL_ADDR_W = (BLOCK_COUNT <= 1) ? 1 : $clog2(BLOCK_COUNT);
  localparam integer DATA_ADDR_W = (BLOCK_COUNT <= 1) ? 1 : $clog2(BLOCK_COUNT);
  localparam integer CTRL_W = `SOFTMAX_M_W + BLOCK_SIZE;
  localparam integer WB_ENTRY_W = `SOFTMAX_M_W + BLOCK_SIZE + BLOCK_SIZE * `SOFTMAX_DESC_W;
  reg [$clog2(BLOCK_COUNT+1)-1:0] blk_wr_count_r;

  wire [WB_ENTRY_W-1:0] entry_word_w;
  wire [BLOCK_SIZE*`SOFTMAX_DESC_W-1:0] entry_desc_block_w;
  wire [BLOCK_SIZE-1:0] entry_mask_w;
  wire signed [`SOFTMAX_M_W-1:0] entry_m_local_w;
  wire entry_has_payload_w;
  wire write_fire_w;

  assign entry_word_w         = entry_i;
  assign entry_desc_block_w   = entry_word_w[BLOCK_SIZE*`SOFTMAX_DESC_W-1:0];
  assign entry_mask_w         = entry_word_w[BLOCK_SIZE*`SOFTMAX_DESC_W+:BLOCK_SIZE];
  assign entry_m_local_w      = entry_word_w[BLOCK_SIZE*`SOFTMAX_DESC_W+BLOCK_SIZE+:`SOFTMAX_M_W];
  assign entry_has_payload_w  = |entry_mask_w;
  assign write_fire_w         = enable_i && entry_valid_i;

  assign idle_o               = !write_fire_w;
  assign blk_wr_count_o       = blk_wr_count_r;

  always @* begin
    ctrl_en_o    = 1'b0;
    ctrl_we_o    = 1'b0;
    ctrl_addr_o  = {CTRL_ADDR_W{1'b0}};
    ctrl_wdata_o = {CTRL_W{1'b0}};
    data_en_o    = 1'b0;
    data_we_o    = 1'b0;
    data_addr_o  = {DATA_ADDR_W{1'b0}};
    data_wdata_o = {BLOCK_SIZE*`SOFTMAX_DESC_W{1'b0}};

    if (write_fire_w) begin
      ctrl_en_o    = 1'b1;
      ctrl_we_o    = 1'b1;
      ctrl_addr_o  = blk_wr_count_r[CTRL_ADDR_W-1:0];
      ctrl_wdata_o = {entry_m_local_w, entry_mask_w};
      data_en_o    = entry_has_payload_w;
      data_we_o    = entry_has_payload_w;
      data_addr_o  = blk_wr_count_r[DATA_ADDR_W-1:0];
      data_wdata_o = entry_desc_block_w;
    end
  end

  always @(posedge clk_i) begin
    if (rst_i) begin
      blk_wr_count_r <= {$clog2(BLOCK_COUNT + 1) {1'b0}};
    end else if (row_start_i) begin
      if (write_fire_w) begin
        blk_wr_count_r <= {{($clog2(BLOCK_COUNT + 1) - 1) {1'b0}}, 1'b1};
      end else begin
        blk_wr_count_r <= {$clog2(BLOCK_COUNT + 1) {1'b0}};
      end
    end else if (write_fire_w) begin
      blk_wr_count_r <= blk_wr_count_r + 1'b1;
    end
  end

endmodule
