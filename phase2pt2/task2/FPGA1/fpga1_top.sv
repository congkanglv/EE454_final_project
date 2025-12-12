// fpga1_top.sv
// FPGA1: compute z = Wx (using linear_layer_q) and capture runtime in cycles
// Show z on LEDR[3:0], show runtime on HEX3..HEX0 (hex), send z on GPIO_Z[3:0]

`timescale 1ns/1ps

module fpga1_top #(
    parameter int IN_DIM      = 4,
    parameter int OUT_DIM     = 1,
    parameter int IN_DATA_W   = 8,
    parameter int OUT_DATA_W  = 4
)(
    input  logic               CLOCK_50,
    input  logic        [3:0]  SW,
    input  logic        [1:0]  KEY,       // KEY[0]=reset (active-low), KEY[1]=start (active-low)

    output logic        [3:0]  LEDR,      // z[3:0]
    output logic               LED_DONE,  // done indicator
    output logic        [3:0]  GPIO_Z,     // z bus to FPGA2

    output logic        [6:0]  HEX0,
    output logic        [6:0]  HEX1,
    output logic        [6:0]  HEX2,
    output logic        [6:0]  HEX3
);

    // ----------------------------
    // Clock, reset, start
    // ----------------------------
    logic clk, rst, start;
    assign clk   = CLOCK_50;
    assign rst   = ~KEY[0];
    assign start = ~KEY[1];

    // ----------------------------
    // Build x_in from switches (0/1 -> 8-bit signed)
    // ----------------------------
    logic signed [IN_DATA_W-1:0] x_in [IN_DIM];

    genvar i;
    generate
        for (i = 0; i < IN_DIM; i++) begin : GEN_X
            assign x_in[i] = {7'b0, SW[i]};
        end
    endgenerate

    // ----------------------------
    // Linear layer
    // ----------------------------
    logic signed [OUT_DATA_W-1:0] z_vec [OUT_DIM];
    logic                         lin_done;

    linear_layer_q #(
        .IN_DIM     (IN_DIM),
        .OUT_DIM    (OUT_DIM),
        .IN_DATA_W  (IN_DATA_W),
        .W_DATA_W   (8),
        .ACC_W      (32),
        .OUT_DATA_W (OUT_DATA_W)
    ) u_lin (
        .clk   (clk),
        .rst   (rst),
        .start (start),
        .x_in  (x_in),
        .y_out (z_vec),
        .done  (lin_done)
    );

    wire [3:0] z = z_vec[0];

    // ----------------------------
    // Runtime capture (cycles)
    // Edge-detect start and done
    // ----------------------------
    logic [15:0] cycle_counter;
    logic [15:0] runtime_cycles;
    logic        measuring;

    logic start_d, done_d;
    wire  start_pulse = start    & ~start_d;   // rising edge
    wire  done_pulse  = lin_done & ~done_d;    // rising edge

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            cycle_counter  <= 16'd0;
            runtime_cycles <= 16'd0;
            measuring      <= 1'b0;
            start_d        <= 1'b0;
            done_d         <= 1'b0;
        end else begin
            // register previous states for edge detect
            start_d <= start;
            done_d  <= lin_done;

            // start measurement on start rising edge
            if (start_pulse) begin
                measuring     <= 1'b1;
                cycle_counter <= 16'd0;
            end else if (measuring && !done_pulse) begin
                // count cycles while measuring (until done edge)
                cycle_counter <= cycle_counter + 16'd1;
            end

            // latch runtime on done rising edge
            if (measuring && done_pulse) begin
                measuring      <= 1'b0;
                runtime_cycles <= cycle_counter;
            end
        end
    end

    // Display live count while running, latched count when finished
    wire [15:0] disp_val = measuring ? cycle_counter : runtime_cycles;

    // ----------------------------
    // Outputs
    // ----------------------------
    assign LEDR     = z;
    assign GPIO_Z   = z;
    assign LED_DONE = lin_done;

    // HEX display (requires hex7seg.sv)
    hex7seg u_hex0(.nibble(disp_val[ 3: 0]), .seg(HEX0));
    hex7seg u_hex1(.nibble(disp_val[ 7: 4]), .seg(HEX1));
    hex7seg u_hex2(.nibble(disp_val[11: 8]), .seg(HEX2));
    hex7seg u_hex3(.nibble(disp_val[15:12]), .seg(HEX3));

endmodule