// mlp_top_q.sv  -- 64 -> 8 -> ReLU -> 10, INT8 weights
`timescale 1ns/1ps

module mlp_top_q #(
    parameter int IN_DIM  = 64,
    parameter int HID_DIM = 8,
    parameter int OUT_DIM = 10
)(
    input  logic                        clk,
    input  logic                        rst,
    input  logic                        start,
    input  logic signed [7:0]           x_in   [IN_DIM],   // int8 input
    output logic signed [31:0]          logits[OUT_DIM],   // int32 output
    output logic [$clog2(OUT_DIM)-1:0]  pred_class,
    output logic                        done
);

    // ========= Layer 1: 64 -> 8, output 32-bit =========
    logic signed [31:0] h_raw[HID_DIM];   // linear output
    logic signed [31:0] h_act[HID_DIM];   // after ReLU
    logic               l1_done;
    logic               l1_start;

    linear_layer_q #(
        .IN_DIM     (IN_DIM),
        .OUT_DIM    (HID_DIM),
        .IN_DATA_W  (8),
        .W_DATA_W   (8),
        .ACC_W      (32),
        .OUT_DATA_W (32),
        .W_FILE     ("W1_q.mem"),
        .B_FILE     ("b1_q.mem")
    ) u_lin1 (
        .clk   (clk),
        .rst   (rst),
        .start (l1_start),
        .x_in  (x_in),
        .y_out (h_raw),
        .done  (l1_done)
    );

    // ReLU (32-bit)
    genvar gi;
    generate
        for (gi = 0; gi < HID_DIM; gi++) begin : GEN_RELU
            relu #(.DATA_W(32)) u_relu (
                .din (h_raw[gi]),
                .dout(h_act[gi])
            );
        end
    endgenerate

    // ========= Layer 2: 8 -> 10, input 32-bit, output 32-bit logits =========
    logic l2_done;
    logic l2_start;

    linear_layer_q #(
        .IN_DIM     (HID_DIM),
        .OUT_DIM    (OUT_DIM),
        .IN_DATA_W  (32),   // 32-bit hidden layer input
        .W_DATA_W   (8),
        .ACC_W      (32),
        .OUT_DATA_W (32),
        .W_FILE     ("W2_q.mem"),
        .B_FILE     ("b2_q.mem")
    ) u_lin2 (
        .clk   (clk),
        .rst   (rst),
        .start (l2_start),
        .x_in  (h_act),
        .y_out (logits),
        .done  (l2_done)
    );

    // ========= Argmax (10 classes) =========
    argmax #(
        .DATA_W(32),
        .DIM   (OUT_DIM)
    ) u_argmax (
        .vec    (logits),
        .idx_max(pred_class)
    );

    // ========= Top-level control FSM =========
    typedef enum logic [1:0] {
        T_IDLE,
        T_L1,
        T_L2,
        T_DONE
    } t_state_t;

    t_state_t t_state;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            t_st_
