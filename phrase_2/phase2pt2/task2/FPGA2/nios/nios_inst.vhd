	component nios is
		port (
			clk_clk                            : in  std_logic                    := 'X';             -- clk
			reset_reset_n                      : in  std_logic                    := 'X';             -- reset_n
			ledr_external_connection_export    : out std_logic_vector(3 downto 0);                    -- export
			gpio_in_external_connection_export : in  std_logic_vector(3 downto 0) := (others => 'X'); -- export
			key_external_connection_export     : in  std_logic                    := 'X'              -- export
		);
	end component nios;

	u0 : component nios
		port map (
			clk_clk                            => CONNECTED_TO_clk_clk,                            --                         clk.clk
			reset_reset_n                      => CONNECTED_TO_reset_reset_n,                      --                       reset.reset_n
			ledr_external_connection_export    => CONNECTED_TO_ledr_external_connection_export,    --    ledr_external_connection.export
			gpio_in_external_connection_export => CONNECTED_TO_gpio_in_external_connection_export, -- gpio_in_external_connection.export
			key_external_connection_export     => CONNECTED_TO_key_external_connection_export      --     key_external_connection.export
		);

