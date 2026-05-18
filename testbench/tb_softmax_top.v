`timescale 1ns/1ps

module tb_softmax_top;

  localparam integer ROW_LEN    = 16;
  localparam integer BLOCK_SIZE = 8;
  localparam integer BLOCK_COUNT = ROW_LEN / BLOCK_SIZE;
  localparam integer TOTAL_ROWS = 32;
  localparam integer TOTAL_BLOCKS = TOTAL_ROWS * BLOCK_COUNT;
  localparam integer FP16_POOL_SIZE = 24;
  localparam integer Y_W = 16;
  localparam integer Y_FRAC_W = 10;
  localparam integer M_W = 7;
  localparam integer DESC_W = 9;
  localparam integer DESC_FRAC_W = 7;
  localparam integer SUM_GLOBAL_W = 16;
  localparam integer EXP_W = DESC_FRAC_W + 1;
  localparam integer OUT_SHIFT = 15 - DESC_FRAC_W;
  localparam integer PAYLOAD_CNT_W = $clog2(BLOCK_SIZE+1);

  reg clk_i;
  reg rst_i;
  reg in_valid_i;
  wire in_ready_o;
  reg [BLOCK_SIZE*16-1:0] in_block_i;
  wire out_valid_o;
  reg out_ready_i;
  wire [BLOCK_SIZE*16-1:0] out_block_o;
  wire out_last_o;

  reg [15:0] input_mem [0:TOTAL_ROWS*ROW_LEN-1];
  reg [BLOCK_SIZE*16-1:0] expected_block [0:TOTAL_BLOCKS-1];
  integer send_row_idx;
  integer send_blk_idx;
  integer recv_block_idx;
  integer error_count;
  integer idx;

  function signed [15:0] sat_q6_10;
    input signed [31:0] value_i;
    begin
      if (value_i > 32'sd32767) begin
        sat_q6_10 = 16'sd32767;
      end else if (value_i < -32'sd32768) begin
        sat_q6_10 = -16'sd32768;
      end else begin
        sat_q6_10 = value_i[15:0];
      end
    end
  endfunction

  function signed [15:0] fp16_to_q6_10;
    input [15:0] fp_i;
    reg sign_bit;
    reg [4:0] exp_field;
    reg [9:0] frac_field;
    reg signed [31:0] scaled_raw;
    integer exp_unbias;
    begin
      sign_bit  = fp_i[15];
      exp_field = fp_i[14:10];
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

  function signed [15:0] m1_convert;
    input [15:0] fp_i;
    reg signed [15:0] x_raw;
    reg signed [31:0] tmp_raw;
    begin
      x_raw = fp16_to_q6_10(fp_i);
      tmp_raw = $signed(x_raw) + ($signed(x_raw) >>> 1) - ($signed(x_raw) >>> 4);
      m1_convert = sat_q6_10(tmp_raw);
    end
  endfunction

  function [EXP_W-1:0] approx_exp_uq1_6;
    input [DESC_FRAC_W-1:0] frac_q6_i;
    reg [EXP_W-1:0] tri_term_q6;
    reg [EXP_W-1:0] corr_q6;
    begin
      if (frac_q6_i[DESC_FRAC_W-1]) begin
        tri_term_q6 = (1 << DESC_FRAC_W) - {1'b0, frac_q6_i};
      end else begin
        tri_term_q6 = {1'b0, frac_q6_i};
      end
      corr_q6 = (tri_term_q6 >> 3) + (tri_term_q6 >> 4);
      approx_exp_uq1_6 = (1 << DESC_FRAC_W) + {1'b0, frac_q6_i} - corr_q6;
    end
  endfunction

  function integer desc_shift_abs;
    input [DESC_W-1:0] desc_i;
    reg [DESC_FRAC_W-1:0] low_bits_i;
    begin
      low_bits_i = desc_i[DESC_FRAC_W-1:0];
      desc_shift_abs = desc_i[DESC_W-1:DESC_FRAC_W] + (|low_bits_i ? 1 : 0);
    end
  endfunction

  function [DESC_FRAC_W-1:0] desc_frac_raw;
    input [DESC_W-1:0] desc_i;
    begin
      desc_frac_raw = {DESC_FRAC_W{1'b0}} - desc_i[DESC_FRAC_W-1:0];
    end
  endfunction

  function [15:0] fp16_pool_word;
    input integer pool_idx;
    begin
      case (pool_idx)
        0:  fp16_pool_word = 16'h0000;
        1:  fp16_pool_word = 16'h0001;
        2:  fp16_pool_word = 16'h3800;
        3:  fp16_pool_word = 16'h3c00;
        4:  fp16_pool_word = 16'h3e00;
        5:  fp16_pool_word = 16'h3f00;
        6:  fp16_pool_word = 16'h4000;
        7:  fp16_pool_word = 16'h4100;
        8:  fp16_pool_word = 16'h4200;
        9:  fp16_pool_word = 16'h4300;
        10: fp16_pool_word = 16'h4400;
        11: fp16_pool_word = 16'hbc00;
        12: fp16_pool_word = 16'hbe00;
        13: fp16_pool_word = 16'hc000;
        14: fp16_pool_word = 16'hc100;
        15: fp16_pool_word = 16'hc200;
        16: fp16_pool_word = 16'hc400;
        17: fp16_pool_word = 16'hc800;
        18: fp16_pool_word = 16'hcc00;
        19: fp16_pool_word = 16'h7c00;
        20: fp16_pool_word = 16'hfc00;
        21: fp16_pool_word = 16'h3a00;
        22: fp16_pool_word = 16'hb800;
        default: fp16_pool_word = 16'h3c00;
      endcase
    end
  endfunction

  function [15:0] row_word;
    input integer row_idx_i;
    input integer lane_idx_i;
    integer pool_idx;
    begin
      case (row_idx_i)
        0: begin
          case (lane_idx_i)
            0: row_word = 16'h4000;
            1: row_word = 16'h3c00;
            2: row_word = 16'h0000;
            3: row_word = 16'hbc00;
            4: row_word = 16'hc000;
            5: row_word = 16'h3e00;
            6: row_word = 16'h3800;
            default: row_word = 16'hb800;
          endcase
        end
        1: begin
          case (lane_idx_i)
            0: row_word = 16'h4200;
            1: row_word = 16'h4100;
            2: row_word = 16'h4000;
            3: row_word = 16'h3f00;
            4: row_word = 16'h3e00;
            5: row_word = 16'h3c00;
            6: row_word = 16'h3800;
            default: row_word = 16'h0000;
          endcase
        end
        2: begin
          row_word = (lane_idx_i < 8) ? (16'h3800 + (lane_idx_i << 9)) : 16'h0000;
        end
        3: begin
          row_word = 16'h4400 - (lane_idx_i << 9);
        end
        4: begin
          row_word = 16'h3c00;
        end
        5: begin
          row_word = (lane_idx_i == 0) ? 16'h4300 : ((lane_idx_i == 1) ? 16'h4200 : 16'hbc00);
        end
        6: begin
          row_word = (lane_idx_i == 0) ? 16'h7c00 : ((lane_idx_i == 1) ? 16'hfc00 : fp16_pool_word((lane_idx_i + 10) % FP16_POOL_SIZE));
        end
        7: begin
          row_word = (lane_idx_i[0] == 1'b0) ? 16'h0000 : 16'h0001;
        end
        8: begin
          row_word = (lane_idx_i < 4) ? 16'h4200 : 16'h3c00;
        end
        9: begin
          row_word = (lane_idx_i < 6) ? 16'h4100 : 16'hc400;
        end
        10: begin
          row_word = (lane_idx_i == 7) ? 16'h4300 : 16'hc000;
        end
        11: begin
          row_word = (lane_idx_i < 2) ? 16'h3f00 : (lane_idx_i < 6 ? 16'h3c00 : 16'hb800);
        end
        default: begin
          pool_idx = (row_idx_i * 7 + lane_idx_i * 5) % FP16_POOL_SIZE;
          row_word = fp16_pool_word(pool_idx);
        end
      endcase
    end
  endfunction

  task build_reference;
    reg signed [15:0] y_row [0:ROW_LEN-1];
    reg signed [M_W-1:0] m_local_ref [0:BLOCK_COUNT-1];
    reg [BLOCK_SIZE-1:0] mask_ref [0:BLOCK_COUNT-1];
    reg [PAYLOAD_CNT_W-1:0] payload_ref [0:BLOCK_COUNT-1];
    reg [BLOCK_SIZE*DESC_W-1:0] desc_ref [0:BLOCK_COUNT-1];
    reg signed [M_W-1:0] m_global_ref;
    reg m_global_valid_ref;
    reg [SUM_GLOBAL_W-1:0] sum_global_ref;
    reg signed [M_W-1:0] m_global_final;
    reg [SUM_GLOBAL_W-1:0] sum_global_final;
    reg signed [15:0] ymax_wide;
    reg signed [M_W-1:0] ymax_floor;
    reg frac_nonzero;
    integer row_idx;
    integer blk_idx;
    integer lane_idx;
    integer near_peak_cnt;
    integer tau_elem;
    integer tau_blk;
    integer delta_blk;
    integer z_q6_raw;
    integer neg_z_q6_raw;
    integer shift_abs;
    integer frac_q6;
    integer sum_local;
    integer merged_exp_q6;
    integer exp_rshift_extra;
    integer final_rshift;
    integer base_rshift;
    integer gap_shift;
    integer norm_msb;
    reg signed [5:0] denom_e;
    reg [DESC_FRAC_W:0] norm_raw;
    reg [DESC_FRAC_W-1:0] denom_frac_q6;
    reg [DESC_FRAC_W-1:0] denom_comp_q6;
    reg [DESC_FRAC_W-1:0] lut_in_q6;
    reg [EXP_W-1:0] exp_term;
    begin
      for (row_idx = 0; row_idx < TOTAL_ROWS; row_idx = row_idx + 1) begin
        for (idx = 0; idx < ROW_LEN; idx = idx + 1) begin
          y_row[idx] = m1_convert(input_mem[row_idx*ROW_LEN + idx]);
        end

        m_global_valid_ref = 1'b0;
        m_global_ref = {M_W{1'b0}};
        sum_global_ref = {SUM_GLOBAL_W{1'b0}};

        for (blk_idx = 0; blk_idx < BLOCK_COUNT; blk_idx = blk_idx + 1) begin
          ymax_wide = y_row[blk_idx*BLOCK_SIZE];
          for (lane_idx = 1; lane_idx < BLOCK_SIZE; lane_idx = lane_idx + 1) begin
            if (y_row[blk_idx*BLOCK_SIZE + lane_idx] > ymax_wide) begin
              ymax_wide = y_row[blk_idx*BLOCK_SIZE + lane_idx];
            end
          end
          ymax_floor = ymax_wide >>> Y_FRAC_W;
          frac_nonzero = |ymax_wide[Y_FRAC_W-1:0];
          m_local_ref[blk_idx] = ymax_floor + (frac_nonzero ? 1 : 0);

          near_peak_cnt = 0;
          for (lane_idx = 0; lane_idx < BLOCK_SIZE; lane_idx = lane_idx + 1) begin
            if (($signed(y_row[blk_idx*BLOCK_SIZE + lane_idx]) >>> Y_FRAC_W) >= (m_local_ref[blk_idx] - 2)) begin
              near_peak_cnt = near_peak_cnt + 1;
            end
          end

          tau_elem = (near_peak_cnt <= ((BLOCK_SIZE - 1) / 2)) ? -2 : -4;
          tau_blk  = -(DESC_FRAC_W + $clog2(BLOCK_SIZE));
          delta_blk = $signed(m_local_ref[blk_idx]) - $signed(m_global_ref);

          mask_ref[blk_idx] = {BLOCK_SIZE{1'b0}};
          payload_ref[blk_idx] = {PAYLOAD_CNT_W{1'b0}};
          desc_ref[blk_idx] = {BLOCK_SIZE*DESC_W{1'b0}};
          sum_local = 0;

          if (!(m_global_valid_ref && (delta_blk <= tau_blk))) begin
            for (lane_idx = 0; lane_idx < BLOCK_SIZE; lane_idx = lane_idx + 1) begin
              if (($signed(y_row[blk_idx*BLOCK_SIZE + lane_idx]) >>> Y_FRAC_W) >= ($signed(m_local_ref[blk_idx]) + tau_elem)) begin
                mask_ref[blk_idx][lane_idx] = 1'b1;
                payload_ref[blk_idx] = payload_ref[blk_idx] + 1'b1;
                z_q6_raw = ($signed(y_row[blk_idx*BLOCK_SIZE + lane_idx]) -
                            ($signed(m_local_ref[blk_idx]) <<< Y_FRAC_W)) >>> (Y_FRAC_W - DESC_FRAC_W);
                if (z_q6_raw >= 0) begin
                  desc_ref[blk_idx][lane_idx*DESC_W +: DESC_W] = {DESC_W{1'b0}};
                end else begin
                  neg_z_q6_raw = -z_q6_raw;
                  if (neg_z_q6_raw >= ((1 << DESC_W) - 1)) begin
                    desc_ref[blk_idx][lane_idx*DESC_W +: DESC_W] = {DESC_W{1'b1}};
                  end else begin
                    desc_ref[blk_idx][lane_idx*DESC_W +: DESC_W] = neg_z_q6_raw[DESC_W-1:0];
                  end
                end
                sum_local = sum_local +
                            (approx_exp_uq1_6(desc_frac_raw(desc_ref[blk_idx][lane_idx*DESC_W +: DESC_W])) >>
                             desc_shift_abs(desc_ref[blk_idx][lane_idx*DESC_W +: DESC_W]));
              end
            end
          end

          if (!m_global_valid_ref) begin
            m_global_ref = m_local_ref[blk_idx];
            sum_global_ref = sum_local;
            m_global_valid_ref = 1'b1;
          end else if (payload_ref[blk_idx] == 0 && (m_global_valid_ref && (delta_blk <= tau_blk))) begin
            m_global_ref = m_global_ref;
            sum_global_ref = sum_global_ref;
          end else if (m_local_ref[blk_idx] >= m_global_ref) begin
            sum_global_ref = (sum_global_ref >> ($signed(m_local_ref[blk_idx]) - $signed(m_global_ref))) + sum_local;
            m_global_ref = m_local_ref[blk_idx];
          end else begin
            sum_global_ref = sum_global_ref + (sum_local >> ($signed(m_global_ref) - $signed(m_local_ref[blk_idx])));
          end
        end

        m_global_final = m_global_ref;
        sum_global_final = sum_global_ref;

        if (sum_global_final < (1 << DESC_FRAC_W)) begin
          denom_e = -1;
          norm_msb = 0;
          norm_raw = sum_global_final << 1;
        end else begin
          norm_msb = 0;
          for (idx = SUM_GLOBAL_W-1; idx >= 0; idx = idx - 1) begin
            if (sum_global_final[idx] && (norm_msb == 0)) begin
              norm_msb = idx;
            end
          end
          denom_e = norm_msb - DESC_FRAC_W;
          norm_raw = sum_global_final >> (norm_msb - DESC_FRAC_W);
        end

        denom_frac_q6 = norm_raw[DESC_FRAC_W-1:0];
        if (norm_raw[DESC_FRAC_W-1]) begin
          denom_comp_q6 = (((1 << DESC_FRAC_W) - {1'b0, norm_raw[DESC_FRAC_W-1:0]}) >> 3) +
                          (((1 << DESC_FRAC_W) - {1'b0, norm_raw[DESC_FRAC_W-1:0]}) >> 4);
        end else begin
          denom_comp_q6 = (norm_raw[DESC_FRAC_W-1:0] >> 3) + (norm_raw[DESC_FRAC_W-1:0] >> 4);
        end

        for (blk_idx = 0; blk_idx < BLOCK_COUNT; blk_idx = blk_idx + 1) begin
          expected_block[row_idx*BLOCK_COUNT + blk_idx] = {BLOCK_SIZE*16{1'b0}};
          for (lane_idx = 0; lane_idx < BLOCK_SIZE; lane_idx = lane_idx + 1) begin
            if (mask_ref[blk_idx][lane_idx]) begin
              shift_abs = desc_shift_abs(desc_ref[blk_idx][lane_idx*DESC_W +: DESC_W]);
              frac_q6   = desc_frac_raw(desc_ref[blk_idx][lane_idx*DESC_W +: DESC_W]);
              gap_shift = $signed(m_global_final) - $signed(m_local_ref[blk_idx]);
              base_rshift = shift_abs + gap_shift + $signed(denom_e);
              merged_exp_q6 = frac_q6 - denom_frac_q6 - denom_comp_q6;
              if (merged_exp_q6 < 0) begin
                lut_in_q6 = merged_exp_q6 + (1 << DESC_FRAC_W);
                exp_rshift_extra = 1;
              end else begin
                lut_in_q6 = merged_exp_q6[DESC_FRAC_W-1:0];
                exp_rshift_extra = 0;
              end
              exp_term = approx_exp_uq1_6(lut_in_q6);
              final_rshift = base_rshift + exp_rshift_extra;
              expected_block[row_idx*BLOCK_COUNT + blk_idx][lane_idx*16 +: 16] =
                (exp_term >> final_rshift) << OUT_SHIFT;
            end
          end
        end
      end
    end
  endtask

  task init_input_mem;
    integer row_idx;
    integer lane_idx;
    begin
      for (row_idx = 0; row_idx < TOTAL_ROWS; row_idx = row_idx + 1) begin
        for (lane_idx = 0; lane_idx < ROW_LEN; lane_idx = lane_idx + 1) begin
          input_mem[row_idx*ROW_LEN + lane_idx] = row_word(row_idx, lane_idx);
        end
      end
    end
  endtask

  task drive_block;
    input integer row_idx_i;
    input integer blk_idx_i;
    integer lane_idx;
    reg [BLOCK_SIZE*16-1:0] block_word;
    begin : drive_block_proc
      block_word = {BLOCK_SIZE*16{1'b0}};
      for (lane_idx = 0; lane_idx < BLOCK_SIZE; lane_idx = lane_idx + 1) begin
        block_word[lane_idx*16 +: 16] = input_mem[row_idx_i*ROW_LEN + blk_idx_i*BLOCK_SIZE + lane_idx];
      end
      in_block_i <= block_word;
      in_valid_i <= 1'b1;
      while (1'b1) begin
        @(posedge clk_i);
        if (in_ready_o) begin
          in_valid_i <= 1'b0;
          in_block_i <= {BLOCK_SIZE*16{1'b0}};
          disable drive_block_proc;
        end
      end
    end
  endtask

  softmax_top #(
    .ROW_LEN(ROW_LEN),
    .BLOCK_SIZE(BLOCK_SIZE)
  ) dut (
    .clk_i(clk_i),
    .rst_i(rst_i),
    .in_valid_i(in_valid_i),
    .in_ready_o(in_ready_o),
    .in_block_i(in_block_i),
    .out_valid_o(out_valid_o),
    .out_ready_i(out_ready_i),
    .out_block_o(out_block_o),
    .out_last_o(out_last_o)
  );

  always #5 clk_i = ~clk_i;

  initial begin
    clk_i = 1'b0;
    rst_i = 1'b1;
    in_valid_i = 1'b0;
    in_block_i = {BLOCK_SIZE*16{1'b0}};
    out_ready_i = 1'b1;
    recv_block_idx = 0;
    error_count = 0;

    init_input_mem;

    build_reference;

    repeat (4) @(posedge clk_i);
    rst_i = 1'b0;
    repeat (2) @(posedge clk_i);

    for (send_row_idx = 0; send_row_idx < TOTAL_ROWS; send_row_idx = send_row_idx + 1) begin
      for (send_blk_idx = 0; send_blk_idx < BLOCK_COUNT; send_blk_idx = send_blk_idx + 1) begin
        drive_block(send_row_idx, send_blk_idx);
      end
    end

    repeat (300) @(posedge clk_i);
    if (recv_block_idx != TOTAL_BLOCKS) begin
      $display("ERROR: timed out before receiving all output blocks, recv=%0d exp=%0d", recv_block_idx, TOTAL_BLOCKS);
      $fatal;
    end
    if (error_count != 0) begin
      $display("ERROR: softmax_top mismatches=%0d", error_count);
      $fatal;
    end
    $display("------------------------------PASS!--------------------------------");
    $finish;
  end

  always @(posedge clk_i) begin
    if (!rst_i && out_valid_o && out_ready_i) begin
      if (out_block_o !== expected_block[recv_block_idx]) begin
        $display("ERROR: block %0d mismatch", recv_block_idx);
        $display("  got=%h", out_block_o);
        $display("  exp=%h", expected_block[recv_block_idx]);
        error_count = error_count + 1;
      end
      if (out_last_o !== ((recv_block_idx % BLOCK_COUNT) == (BLOCK_COUNT - 1))) begin
        $display("ERROR: out_last mismatch on block %0d", recv_block_idx);
        error_count = error_count + 1;
      end
      recv_block_idx = recv_block_idx + 1;
    end
  end

endmodule
