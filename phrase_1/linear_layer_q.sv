// linear_layer_q.sv  -- INT8 weights + signed fixed-point linear layer
`timescale 1ns/1ps

module linear_layer_q #(
    parameter int IN_DIM      = 64,
    parameter int OUT_DIM     = 8,
    parameter int IN_DATA_W   = 8,   // input bit-width
    parameter int W_DATA_W    = 8,   // weight bit-width
    parameter int ACC_W       = 32,  // accumulator bit-width
    parameter int OUT_DATA_W  = 32,  // output bit-width
    parameter string W_FILE   = "W1_q.mem", // weight file
    parameter string B_FILE   = "b1_q.mem"  // bias file
)(
    input  logic                          clk,
    input  logic                          rst,
    input  logic                          start,
    input  logic signed [IN_DATA_W-1:0]   x_in [IN_DIM],
    output logic signed [OUT_DATA_W-1:0]  y_out[OUT_DIM],
    output logic                          done
);

    // --------- Weights & Biases ---------
    reg  signed [W_DATA_W-1:0]  W_q [0:IN_DIM*OUT_DIM-1];
    reg  signed [ACC_W-1:0]     B_q [0:OUT_DIM-1];

    initial begin
        $readmemh(W_FILE, W_q);
        $readmemh(B_FILE, B_q);
    end

    // --------- State Machine ---------
    typedef enum logic [1:0] {
        S_IDLE,
        S_MAC,
        S_DONE
    } state_t;

    state_t state;

    integer out_idx;   // current output neuron index
    integer in_idx;    // current input index

    // Accumulator and multiply logic
    logic signed [ACC_W-1:0] acc;
    logic signed [ACC_W-1:0] mult_full;
    logic signed [ACC_W-1:0] acc_next;
    integer w_index;

    // --------- Saturation function (truncate from ACC_W to OUT_DATA_W) ---------
    function [OUT_DATA_W-1:0] sat_narrow;
        input signed [ACC_W-1:0] v;

        integer max_v_int, min_v_int;
        integer tmp;
        reg [OUT_DATA_W-1:0] y;
    begin
        max_v_int = (1 <<< (OUT_DATA_W-1)) - 1;
        min_v_int = - (1 <<< (OUT_DATA_W-1));

        tmp = v;  // safe when ACC_W <= 32

        if (tmp > max_v_int)
            tmp = max_v_int;
        else if (tmp < min_v_int)
            tmp = min_v_int;

        y = tmp[OUT_DATA_W-1:0];
        sat_narrow = y;
    end
    endfunction

    // --------- Combinational logic: multiply + accumulate ---------
    always_comb begin
        w_index   = in_idx * OUT_DIM + out_idx;
        // multiplication result extended into ACC_W bit-width (higher bits truncated)
        mult_full = x_in[in_idx] * W_q[w_index];
        acc_next  = acc + mult_full;
    end

    // --------- Main State Machine ---------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state   <= S_IDLE;
            out_idx <= 0;
            in_idx  <= 0;
            acc     <= '0;
            done    <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        out_idx <= 0;
                        in_idx  <= 0;
                        acc     <= B_q[0];   // initialize accumulator with bias
                        state   <= S_MAC;
                    end
                end

                S_MAC: begin
                    acc <= acc_next;

                    if (in_idx == IN_DIM-1) begin
                        // finished computing for the current neuron, write result
                        y_out[out_idx] <= sat_narrow(acc_next);

                        if (out_idx == OUT_DIM-1) begin
                            state <= S_DONE;
                        end else begin
                            out_idx <= out_idx + 1;
                            in_idx  <= 0;
                            acc     <= B_q[out_idx + 1];
                        end
                    end else begin
                        in_idx <= in_idx + 1;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    if (!start) begin
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
