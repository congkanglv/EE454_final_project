// relu.sv
`timescale 1ns/1ps

module relu #(
    parameter int DATA_W = 8
)(
    input  logic signed [DATA_W-1:0] din,
    output logic signed [DATA_W-1:0] dout
);
    always_comb begin
        if (din[DATA_W-1] == 1'b1)    // negative number -> output 0
            dout = '0;
        else
            dout = din;
    end
endmodule