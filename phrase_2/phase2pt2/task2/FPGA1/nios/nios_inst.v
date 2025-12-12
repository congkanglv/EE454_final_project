	nios u0 (
		.clk_clk                             (<connected-to-clk_clk>),                             //                          clk.clk
		.reset_reset_n                       (<connected-to-reset_reset_n>),                       //                        reset.reset_n
		.led_done_external_connection_export (<connected-to-led_done_external_connection_export>), // led_done_external_connection.export
		.ledr_external_connection_export     (<connected-to-ledr_external_connection_export>),     //     ledr_external_connection.export
		.sw_external_connection_export       (<connected-to-sw_external_connection_export>),       //       sw_external_connection.export
		.gpio_z_external_connection_export   (<connected-to-gpio_z_external_connection_export>),   //   gpio_z_external_connection.export
		.key_external_connection_export      (<connected-to-key_external_connection_export>)       //      key_external_connection.export
	);

