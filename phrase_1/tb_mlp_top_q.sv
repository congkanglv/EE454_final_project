// tb_mlp_top_q_simple.sv  -- Simple testbench: wait sufficient time and print results
`timescale 1ns/1ps

module tb_mlp_top_q_simple;

    // Clock / Reset / Control
    logic clk;
    logic rst;
    logic start;

    // DUT ports
    logic  signed [7:0]   x_in   [64];
    logic  signed [31:0]  logits [10];
    logic  [3:0]          pred_class;
    logic                 done;

    // Instantiate DUT
    mlp_top_q #(
        .IN_DIM (64),
        .HID_DIM(8),
        .OUT_DIM(10)
    ) dut (
        .clk        (clk),
        .rst        (rst),
        .start      (start),
        .x_in       (x_in),
        .logits     (logits),
        .pred_class (pred_class),
        .done       (done)
    );

    // 100 MHz clock
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // Load input vector
    initial begin
        $display("TB: Loading x_sample.mem ...");
        $readmemh("x_sample.mem", x_in);
    end

    integer i;

    // Main stimulus: generate a start pulse, wait 20us, then sample outputs
    initial begin
        rst   = 1'b1;
        start = 1'b0;

        #50;
        rst = 1'b0;

        #50;

        $display("TB: Asserting start...");
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;

        // Wait sufficient time (> 6us)
        #20000;  // 20us

        $display("TB: Sampling outputs after 20us...");

        for (i = 0; i < 10; i = i + 1) begin
            $display("logits[%0d] = %0d", i, logits[i]);
        end

        $display("pred_class = %0d", pred_class);
        $display("NOTE: Python quantized model gave argmax = 0 for this sample.");
        $display("      If pred_class == 0, then HW & SW predictions match.");

        #20;
        $finish;
    end

endmodule
