// fpga1_top.sv -- FPGA1: compute z = Wx on KEY1 press and send 4-bit z to FPGA2

module fpga1_top #(
    parameter int IN_DIM      = 4,
    parameter int OUT_DIM     = 1,
    parameter int IN_DATA_W   = 8,
    parameter int OUT_DATA_W  = 4
)(
    input  logic               CLOCK_50,  // 50 MHz clock
    input  logic        [3:0]  SW,        // 4 switches as inputs
    input  logic        [1:0]  KEY,       // KEY[0]=reset (active-low), KEY[1]=start (active-low)
    output logic        [3:0]  LEDR,      // show z[3:0]
    output logic               LED_DONE,  // done indicator
    output logic        [3:0]  GPIO_Z     // 4-bit bus to FPGA2
);

    // Clock, reset, start
    logic clk;
    logic rst;
    logic start;

    assign clk   = CLOCK_50;
    assign rst   = ~KEY[0];   // press KEY0 => rst = 1
    assign start = ~KEY[1];   // press KEY1 => start = 1

    // Input vector x_in[0..3] from switches (0 or 1 extended to 8 bits)
    logic signed [IN_DATA_W-1:0] x_in [IN_DIM];

    genvar i;
    generate
        for (i = 0; i < IN_DIM; i++) begin : GEN_X
            assign x_in[i] = {7'b0, SW[i]};
        end
    endgenerate

    // Output from linear layer
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

    // Single neuron output
    wire [3:0] z = z_vec[0];

    assign LEDR     = z;
    assign GPIO_Z   = z;
    assign LED_DONE = lin_done;

endmodule


/*
module fpga1_top (
    input  logic        CLOCK_50,
    input  logic [3:0]  SW,
    input  logic [1:0]  KEY,
    output logic [3:0]  LEDR,
    output logic        LED_DONE,
    output logic [3:0]  GPIO_Z
);
    // TEMPORARY DEBUG: LEDs = switches
    assign LEDR     = SW;
    assign LED_DONE = 1'b0;
    assign GPIO_Z   = 4'b0000;
endmodule
*/