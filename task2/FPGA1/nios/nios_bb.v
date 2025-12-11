
module nios (
	clk_clk,
	reset_reset_n,
	led_done_external_connection_export,
	ledr_external_connection_export,
	sw_external_connection_export,
	gpio_z_external_connection_export,
	key_external_connection_export);	

	input		clk_clk;
	input		reset_reset_n;
	output		led_done_external_connection_export;
	output	[3:0]	ledr_external_connection_export;
	input	[3:0]	sw_external_connection_export;
	output	[3:0]	gpio_z_external_connection_export;
	input	[1:0]	key_external_connection_export;
endmodule
