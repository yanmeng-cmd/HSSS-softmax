`include "define.vh"

module softmax_row_slot #(
    parameter integer ROW_LEN    = `SOFTMAX_ROW_LEN,
    parameter integer BLOCK_SIZE = `SOFTMAX_BLOCK_SIZE
) (
    input wire clk_i,
    input wire rst_i,
    input wire ctrl_en_i,
    input wire ctrl_we_i,
    input wire [((ROW_LEN / BLOCK_SIZE) <= 1 ? 1 : $clog2(ROW_LEN / BLOCK_SIZE))-1:0] ctrl_addr_i,
    input wire [`SOFTMAX_M_W + BLOCK_SIZE-1:0] ctrl_wdata_i,
    output reg [`SOFTMAX_M_W + BLOCK_SIZE-1:0] ctrl_rdata_o,
    input wire data_en_i,
    input wire data_we_i,
    input wire [((ROW_LEN / BLOCK_SIZE) <= 1 ? 1 : $clog2(ROW_LEN / BLOCK_SIZE))-1:0] data_addr_i,
    input wire [BLOCK_SIZE*`SOFTMAX_DESC_W-1:0] data_wdata_i,
    output reg [BLOCK_SIZE*`SOFTMAX_DESC_W-1:0] data_rdata_o,
    input wire row_ctx_we_i,
    input wire [`SOFTMAX_M_W + `SOFTMAX_SUM_GLOBAL_W - 1:0] row_ctx_wdata_i,
    output reg [`SOFTMAX_M_W + `SOFTMAX_SUM_GLOBAL_W - 1:0] row_ctx_rdata_o
);

  localparam integer BLOCK_COUNT = ROW_LEN / BLOCK_SIZE;
  localparam integer CTRL_ADDR_W = (BLOCK_COUNT <= 1) ? 1 : $clog2(BLOCK_COUNT);
  localparam integer DATA_ADDR_W = (BLOCK_COUNT <= 1) ? 1 : $clog2(BLOCK_COUNT);
  localparam integer CTRL_W = `SOFTMAX_M_W + BLOCK_SIZE;
  localparam integer ROW_CTX_W = `SOFTMAX_M_W + `SOFTMAX_SUM_GLOBAL_W;

  (* ram_style = "distributed" *) reg [CTRL_W-1:0] ctrl_ram_r[0:BLOCK_COUNT-1];
  (* ram_style = "block" *) reg [BLOCK_SIZE*`SOFTMAX_DESC_W-1:0] data_ram_r[0:BLOCK_COUNT-1];
  wire [CTRL_W-1:0] ctrl_rdata_raw_w;

  assign ctrl_rdata_raw_w = ctrl_ram_r[ctrl_addr_i];

  always @(posedge clk_i) begin
    if (rst_i) begin
      ctrl_rdata_o <= {CTRL_W{1'b0}};
    end else begin
      if (ctrl_en_i && ctrl_we_i) begin
        ctrl_ram_r[ctrl_addr_i] <= ctrl_wdata_i;
      end
      if (ctrl_en_i) begin
        ctrl_rdata_o <= ctrl_rdata_raw_w;
      end
    end
  end

  always @(posedge clk_i) begin
    if (rst_i) begin
      data_rdata_o <= {BLOCK_SIZE * `SOFTMAX_DESC_W{1'b0}};
    end else begin
      if (data_en_i) begin
        if (data_we_i) begin
          data_ram_r[data_addr_i] <= data_wdata_i;
        end
        data_rdata_o <= data_ram_r[data_addr_i];
      end
    end
  end

  always @(posedge clk_i) begin
    if (rst_i) begin
      row_ctx_rdata_o <= {ROW_CTX_W{1'b0}};
    end else begin
      if (row_ctx_we_i) begin
        row_ctx_rdata_o <= row_ctx_wdata_i;
      end
    end
  end

endmodule
