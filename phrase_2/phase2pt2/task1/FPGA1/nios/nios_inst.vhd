	component nios is
		port (
			clk_clk                             : in  std_logic := 'X'; -- clk
			reset_reset_n                       : in  std_logic := 'X'; -- reset_n
			gpio_out_external_connection_export : out std_logic;        -- export
			switch_external_connection_export   : in  std_logic := 'X'  -- export
		);
	end component nios;

	u0 : component nios
		port map (
			clk_clk                             => CONNECTED_TO_clk_clk,                             --                          clk.clk
			reset_reset_n                       => CONNECTED_TO_reset_reset_n,                       --                        reset.reset_n
			gpio_out_external_connection_export => CONNECTED_TO_gpio_out_external_connection_export, -- gpio_out_external_connection.export
			switch_external_connection_export   => CONNECTED_TO_switch_external_connection_export    --   switch_external_connection.export
		);

