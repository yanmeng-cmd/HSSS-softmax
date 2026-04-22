`include "define.vh"

module softmax_m5_online_update #(
    parameter integer BLOCK_SIZE = `SOFTMAX_BLOCK_SIZE
) (
    input  wire                                                       clk_i,
    input  wire                                                       rst_i,
    input  wire                                                       block_fire_i,
    input  wire                                                       row_start_i,
    input  wire                                                       row_last_i,
    input  wire                                                       block_prune_i,
    input  wire signed [                            `SOFTMAX_M_W-1:0] m_local_i,
    input  wire        [$clog2(BLOCK_SIZE+1)+`SOFTMAX_SUM_FRAC_W-1:0] sum_local_i,
    output reg signed  [                            `SOFTMAX_M_W-1:0] m_global_o,
    output reg                                                        m_global_valid_o,
    output reg signed  [                            `SOFTMAX_M_W-1:0] m_global_next_o,
    output reg         [                   `SOFTMAX_SUM_GLOBAL_W-1:0] sum_global_next_o,
    output wire                                                       row_done_o
);

  localparam integer SUM_LOCAL_W = $clog2(BLOCK_SIZE + 1) + `SOFTMAX_SUM_FRAC_W;
  localparam integer SHIFT_W = $clog2(`SOFTMAX_SUM_GLOBAL_W + 1);
  localparam integer DELTA_W = `SOFTMAX_M_W + 1;
  localparam [SHIFT_W-1:0] MAX_SHIFT = `SOFTMAX_SUM_GLOBAL_W;

  function [`SOFTMAX_SUM_GLOBAL_W-1:0] shift_right_clip;
    input [`SOFTMAX_SUM_GLOBAL_W-1:0] value_i;
    input [SHIFT_W-1:0] shift_i;
    begin
      if (shift_i >= MAX_SHIFT) begin
        shift_right_clip = {`SOFTMAX_SUM_GLOBAL_W{1'b0}};
      end else begin
        shift_right_clip = value_i >> shift_i;
      end
    end
  endfunction

  reg [`SOFTMAX_SUM_GLOBAL_W-1:0] sum_global_r;
  reg [`SOFTMAX_SUM_GLOBAL_W-1:0] sum_local_ext;
  reg [`SOFTMAX_SUM_GLOBAL_W-1:0] shifted_old;
  reg [`SOFTMAX_SUM_GLOBAL_W-1:0] shifted_new;
  reg [SHIFT_W-1:0] delta_shift;
  reg [DELTA_W-1:0] delta_abs;
  reg use_new_max;
  reg signed [DELTA_W-1:0] m_local_ext;
  reg signed [DELTA_W-1:0] m_global_ext;

  assign row_done_o = block_fire_i && row_last_i;

  always @* begin
    sum_local_ext     = {{(`SOFTMAX_SUM_GLOBAL_W - SUM_LOCAL_W) {1'b0}}, sum_local_i};
    shifted_old       = {`SOFTMAX_SUM_GLOBAL_W{1'b0}};
    shifted_new       = {`SOFTMAX_SUM_GLOBAL_W{1'b0}};
    delta_shift       = {SHIFT_W{1'b0}};
    delta_abs         = {DELTA_W{1'b0}};
    use_new_max       = (m_local_i >= m_global_o);
    m_local_ext       = {m_local_i[`SOFTMAX_M_W-1], m_local_i};
    m_global_ext      = {m_global_o[`SOFTMAX_M_W-1], m_global_o};
    m_global_next_o   = m_global_o;
    sum_global_next_o = sum_global_r;

    if (use_new_max) begin
      delta_abs = m_local_ext - m_global_ext;
    end else begin
      delta_abs = m_global_ext - m_local_ext;
    end

    if (delta_abs >= MAX_SHIFT) begin
      delta_shift = MAX_SHIFT;
    end else begin
      delta_shift = delta_abs[SHIFT_W-1:0];
    end

    shifted_old = shift_right_clip(sum_global_r, delta_shift);
    shifted_new = shift_right_clip(sum_local_ext, delta_shift);

    if (row_start_i || !m_global_valid_o) begin
      m_global_next_o   = m_local_i;
      sum_global_next_o = sum_local_ext;
    end else if (block_prune_i) begin
      m_global_next_o   = m_global_o;
      sum_global_next_o = sum_global_r;
    end else if (use_new_max) begin
      m_global_next_o   = m_local_i;
      sum_global_next_o = shifted_old + sum_local_ext;
    end else begin
      m_global_next_o   = m_global_o;
      sum_global_next_o = sum_global_r + shifted_new;
    end
  end

  always @(posedge clk_i) begin
    if (rst_i) begin
      m_global_o       <= {`SOFTMAX_M_W{1'b0}};
      m_global_valid_o <= 1'b0;
      sum_global_r     <= {`SOFTMAX_SUM_GLOBAL_W{1'b0}};
    end else if (block_fire_i) begin
      if (row_last_i) begin
        m_global_o       <= {`SOFTMAX_M_W{1'b0}};
        m_global_valid_o <= 1'b0;
        sum_global_r     <= {`SOFTMAX_SUM_GLOBAL_W{1'b0}};
      end else begin
        m_global_o       <= m_global_next_o;
        m_global_valid_o <= 1'b1;
        sum_global_r     <= sum_global_next_o;
      end
    end
  end

endmodule
