
module nios (
	clk_clk,
	reset_reset_n,
	switches_external_connection_export,
	leds_external_connection_export);	

	input		clk_clk;
	input		reset_reset_n;
	input	[7:0]	switches_external_connection_export;
	output	[7:0]	leds_external_connection_export;
endmodule
