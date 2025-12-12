// fpga2_top.sv
// Board 2: y = ReLU(z + b), 4-bit fixed-point
`timescale 1ns/1ps

module fpga2_top #(
    parameter int DATA_W = 4
)(
    input  logic               CLOCK_50,   // not used, but kept as a port
    input  logic        [0:0]  KEY,        // not used, but kept as a port
    input  logic [DATA_W-1:0]  GPIO_IN,    // 4-bit z from FPGA1
    output logic [DATA_W-1:0]  LEDR        // show y on LEDs
);

    // Treat GPIO_IN as signed 4-bit z
    logic signed [DATA_W-1:0] z;
    logic signed [DATA_W-1:0] b;
    logic signed [DATA_W-1:0] sum;
    logic signed [DATA_W-1:0] relu_out;

    assign z = GPIO_IN;

    // Example bias: +1 (range -8..7 for 4-bit signed)
    initial b = 4'sd1;

    assign sum = z + b;

    // Your existing relu module:
    // module relu #(parameter DATA_W = 32)(
    //   input  logic signed [DATA_W-1:0] din,
    //   output logic signed [DATA_W-1:0] dout);
    relu #(
        .DATA_W(DATA_W)
    ) u_relu (
        .din  (sum),
        .dout (relu_out)
    );

    // Drive LEDs combinationally with ReLU output
    always_comb begin
        LEDR = relu_out[DATA_W-1:0];
    end

endmodule