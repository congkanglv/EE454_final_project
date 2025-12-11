// argmax.sv
`timescale 1ns/1ps

module argmax #(
    parameter int DATA_W = 32,
    parameter int DIM    = 10
)(
    input  logic signed [DATA_W-1:0] vec [DIM],
    output logic [$clog2(DIM)-1:0]   idx_max
);
    integer k;
    logic signed [DATA_W-1:0] cur_max;
    logic [$clog2(DIM)-1:0]   cur_idx;

    always_comb begin
        cur_max = vec[0];
        cur_idx = '0;

        for (k = 1; k < DIM; k++) begin
            if (vec[k] > cur_max) begin
                cur_max = vec[k];
                cur_idx = k[$clog2(DIM)-1:0];
            end
        end

        idx_max = cur_idx;
    end
endmodule
