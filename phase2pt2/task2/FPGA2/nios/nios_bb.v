
module nios (
	clk_clk,
	reset_reset_n,
	ledr_external_connection_export,
	gpio_in_external_connection_export,
	key_external_connection_export);	

	input		clk_clk;
	input		reset_reset_n;
	output	[3:0]	ledr_external_connection_export;
	input	[3:0]	gpio_in_external_connection_export;
	input		key_external_connection_export;
endmodule
