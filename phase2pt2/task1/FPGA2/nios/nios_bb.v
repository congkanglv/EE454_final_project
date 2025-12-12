
module nios (
	clk_clk,
	reset_reset_n,
	led_external_connection_export,
	gpio_in_external_connection_export);	

	input		clk_clk;
	input		reset_reset_n;
	output		led_external_connection_export;
	input		gpio_in_external_connection_export;
endmodule
